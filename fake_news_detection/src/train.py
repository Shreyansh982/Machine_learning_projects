import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_io import save_model
from preprocess import (
    ARTIFACT_DIR, CLASS_NAMES, CLASS_ORDER, DATA_DIR, FEATURE_COLUMNS,
    FEATURE_VERSION, LABEL_MAPPING, NUMERIC_FEATURES, POSITIVE_CLASS,
    load_and_process_data,
)

EXPERIMENT_NAME = 'LIAR_Truthfulness_RF'
EXPECTED_ROWS = {'train': 10269, 'valid': 1284, 'test': 1283}


def build_pipeline(n_estimators=200):
    preprocessor = ColumnTransformer([
        ('text', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)), 'combined_text'),
        ('num', StandardScaler(), NUMERIC_FEATURES),
    ])
    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=n_estimators, max_depth=None, random_state=42,
            n_jobs=-1, class_weight='balanced',
        )),
    ])


def evaluate(model, features, targets, dataset_name, output_dir):
    predictions = model.predict(features)
    names = [CLASS_NAMES[label] for label in CLASS_ORDER]
    report = classification_report(
        targets, predictions, labels=CLASS_ORDER, target_names=names,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=CLASS_ORDER)
    metrics = {'accuracy': float((predictions == targets).mean())}
    for label, name in CLASS_NAMES.items():
        for metric in ('precision', 'recall', 'f1-score', 'support'):
            metrics[f"class_{label}_{metric.replace('-', '_')}"] = report[name][metric]
    metrics['macro_f1'] = report['macro avg']['f1-score']
    metrics['weighted_f1'] = report['weighted avg']['f1-score']
    metrics['binary_f1_lower_truthfulness'] = f1_score(
        targets, predictions, average='binary', pos_label=POSITIVE_CLASS, zero_division=0,
    )
    result = {
        'dataset': dataset_name, 'class_order': CLASS_ORDER, 'class_names': names,
        'classification_report': report, 'confusion_matrix': matrix.tolist(), 'metrics': metrics,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f'{dataset_name}_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    text_report = classification_report(
        targets, predictions, labels=CLASS_ORDER, target_names=names, zero_division=0, digits=4,
    )
    (output_dir / f'{dataset_name}_classification_report.txt').write_text(text_report, encoding='utf-8')
    print(f'\n{dataset_name.title()} evaluation\n{text_report}')
    if dataset_name == 'test':
        fig, ax = plt.subplots(figsize=(9, 7))
        ConfusionMatrixDisplay(matrix, display_labels=names).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set(xlabel='Predicted class', ylabel='Actual class', title='LIAR held-out test set - confusion matrix')
        fig.tight_layout()
        fig.savefig(output_dir / 'confusion_matrix.png', dpi=150)
        plt.close(fig)
    return result


def configure_mlflow(output_dir):
    tracking_dir = Path(output_dir).resolve() / 'tracking'
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(tracking_dir / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(tracking_dir / 'mlruns').as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def train_model(data_dir=DATA_DIR, output_dir=ARTIFACT_DIR, expected_rows=EXPECTED_ROWS, n_estimators=200):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    splits = {
        split: load_and_process_data(data_dir / f'{split}.tsv', expected_rows=expected_rows[split])
        for split in EXPECTED_ROWS
    }
    for first, second in (('train', 'valid'), ('train', 'test'), ('valid', 'test')):
        if set(splits[first]['id']) & set(splits[second]['id']):
            raise ValueError(f'Overlapping claim IDs between {first} and {second} splits.')
    if set(splits['train']['target']) != set(CLASS_ORDER):
        raise ValueError('Training data must contain both target classes.')
    configure_mlflow(output_dir)
    with mlflow.start_run() as run:
        model = build_pipeline(n_estimators)
        params = {f'rf_{key}': value for key, value in model['clf'].get_params().items()}
        params.update({'tfidf_max_features': 5000, 'tfidf_ngram_range': '(1, 2)',
                       'tfidf_stop_words': 'english', 'feature_version': FEATURE_VERSION,
                       'positive_class': POSITIVE_CLASS, 'model_selection': 'fixed configuration; no search'})
        mlflow.log_params(params)
        provenance = {'label_mapping': LABEL_MAPPING, 'class_names': CLASS_NAMES,
                      'feature_columns': FEATURE_COLUMNS, 'feature_version': FEATURE_VERSION,
                      'run_id': run.info.run_id, 'datasets': {}}
        for split, frame in splits.items():
            provenance['datasets'][split] = {
                'file': f'{split}.tsv', 'rows': len(frame),
                'sha256': hashlib.sha256((data_dir / f'{split}.tsv').read_bytes()).hexdigest(),
                'missing_values': frame.isna().sum().to_dict(),
                'history_zero_inconsistencies': frame.attrs['history_zero_inconsistencies'],
            }
        print('Training fixed Random Forest configuration on the training split only...')
        model.fit(splits['train'][FEATURE_COLUMNS], splits['train']['target'])
        results = {}
        # Configuration is fixed in advance; validation and test never fit any pipeline step.
        for split in ('valid', 'test'):
            results[split] = evaluate(model, splits[split][FEATURE_COLUMNS], splits[split]['target'], split, output_dir)
            mlflow.log_metrics({f'{split}_{key}': value for key, value in results[split]['metrics'].items()})
            mlflow.log_artifact(str(output_dir / f'{split}_metrics.json'), 'evaluation')
            mlflow.log_artifact(str(output_dir / f'{split}_classification_report.txt'), 'evaluation')
        mlflow.log_artifact(str(output_dir / 'confusion_matrix.png'), 'evaluation')
        model_info = mlflow.sklearn.log_model(
            sk_model=model, name='model', input_example=splits['train'][FEATURE_COLUMNS].head(2),
            pip_requirements=[f'{name}=={version(name)}'
                              for name in ('scikit-learn', 'numpy', 'scipy', 'pandas', 'cloudpickle')],
        )
        provenance['mlflow_model_uri'] = model_info.model_uri
        (output_dir / 'training_metadata.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        mlflow.log_artifact(str(output_dir / 'training_metadata.json'))
        save_model(model, output_dir / 'model.joblib', provenance)
        print(f"Model: {output_dir / 'model.joblib'}\nRun ID: {run.info.run_id}\nModel URI: {model_info.model_uri}")
        print(f"Confusion matrix: {output_dir / 'confusion_matrix.png'}")
    return model, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate the fixed LIAR Random Forest pipeline.')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR, help='Directory containing the original three LIAR TSV splits.')
    parser.add_argument('--output-dir', type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    train_model(args.data_dir, args.output_dir)
