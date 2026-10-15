import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

import model_io
import preprocess
import train
from model_io import load_model, predict_probabilities, save_model
from preprocess import (
    CLASS_NAMES, CLASS_ORDER, COLUMNS, FEATURE_COLUMNS, HISTORY_COLUMNS,
    LABEL_MAPPING, NUMERIC_FEATURES, TEXT_METADATA, load_and_process_data, prepare_features,
)


def test_label_mapping_and_history_excludes_current_claim(tmp_path, raw_claims, write_tsv):
    data = load_and_process_data(write_tsv(tmp_path / 'claims.tsv', raw_claims), expected_rows=6)
    assert data['target'].tolist() == [1, 1, 1, 0, 0, 0]
    for _, row in data.iterrows():
        for label, column in HISTORY_COLUMNS.items():
            assert row[column] == (2 if row['label'] == label else 3)
    assert data['statement'][0] == raw_claims['statement'][0]


@pytest.mark.parametrize('label', ['', 'unknown', 'FALSE', 'nan'])
def test_invalid_labels_rejected(tmp_path, raw_claims, write_tsv, label):
    raw_claims.loc[0, 'label'] = label
    with pytest.raises(ValueError, match='invalid or missing labels'):
        load_and_process_data(write_tsv(tmp_path / 'bad.tsv', raw_claims))


@pytest.mark.parametrize('contents', ['', 'id\tfalse\tstatement\n', '\t'.join(COLUMNS)])
def test_schema_rejected(tmp_path, contents):
    path = tmp_path / 'bad.tsv'
    path.write_text(contents, encoding='utf-8')
    with pytest.raises(ValueError):
        load_and_process_data(path)


def test_wrong_row_count_and_duplicate_ids(tmp_path, raw_claims, write_tsv):
    path = write_tsv(tmp_path / 'claims.tsv', raw_claims)
    with pytest.raises(ValueError, match='expected 7 rows'):
        load_and_process_data(path, expected_rows=7)
    raw_claims.loc[0, 'id'] = raw_claims.loc[1, 'id']
    with pytest.raises(ValueError, match='duplicate claim IDs'):
        load_and_process_data(write_tsv(path, raw_claims))


@pytest.mark.parametrize('value', ['', None, 'invalid', '-1', '1.5', 'inf', '-inf', 'nan'])
def test_bad_numeric_values_rejected(raw_claims, value):
    raw_claims[NUMERIC_FEATURES[0]] = raw_claims[NUMERIC_FEATURES[0]].astype(object)
    raw_claims.loc[0, NUMERIC_FEATURES[0]] = value
    with pytest.raises(ValueError, match='unknown is not zero'):
        prepare_features(raw_claims)


def test_numeric_strings_zeros_and_optional_metadata(raw_claims):
    raw_claims[NUMERIC_FEATURES] = raw_claims[NUMERIC_FEATURES].astype(str)
    raw_claims.loc[0, NUMERIC_FEATURES] = '0'
    raw_claims.loc[0, TEXT_METADATA] = None
    features = prepare_features(raw_claims)
    assert features.columns.tolist() == FEATURE_COLUMNS
    assert all(features[column].dtype == np.float64 for column in NUMERIC_FEATURES)
    assert features.loc[0, NUMERIC_FEATURES].eq(0).all()
    assert features['combined_text'][0].startswith('none none none none ')


@pytest.mark.parametrize('statement', ['', '   ', None, 123])
def test_missing_statement(raw_claims, statement):
    raw_claims.loc[0, 'statement'] = statement
    with pytest.raises(ValueError, match='nonempty text statement'):
        prepare_features(raw_claims)


def test_missing_feature_column(raw_claims):
    with pytest.raises(ValueError, match='Missing feature columns'):
        prepare_features(raw_claims.drop(columns=NUMERIC_FEATURES[0]))


def test_history_zero_inconsistency_is_explicit(tmp_path, raw_claims, write_tsv):
    raw_claims.loc[0, HISTORY_COLUMNS['pants-fire']] = 0
    with pytest.warns(UserWarning, match='already zero'):
        data = load_and_process_data(write_tsv(tmp_path / 'claims.tsv', raw_claims))
    assert data.loc[0, HISTORY_COLUMNS['pants-fire']] == 0
    assert data.attrs['history_zero_inconsistencies'] == 1


def test_training_inference_consistency_and_cwd(tmp_path, raw_claims, write_tsv, monkeypatch):
    project = tmp_path / 'project'
    (project / 'data').mkdir(parents=True)
    write_tsv(project / 'data' / 'test.tsv', raw_claims)
    monkeypatch.setattr(preprocess, 'PROJECT_ROOT', project)
    monkeypatch.chdir(tmp_path)
    loaded = load_and_process_data('data/test.tsv')
    manual = loaded[['statement', *TEXT_METADATA, *NUMERIC_FEATURES]].copy()
    pd.testing.assert_frame_equal(prepare_features(manual), loaded[FEATURE_COLUMNS])
    with pytest.raises(FileNotFoundError, match='Dataset not found'):
        load_and_process_data('data/absent.tsv')


def test_probabilities_follow_model_classes(raw_claims):
    features = prepare_features(raw_claims)
    model = SimpleNamespace(classes_=np.array([1, 0]), predict_proba=lambda frame: np.tile([.8, .2], (len(frame), 1)))
    probabilities = predict_probabilities(model, features)
    np.testing.assert_allclose(probabilities[1], .8)
    np.testing.assert_allclose(probabilities[0], .2)
    model.classes_ = np.array([0, 2])
    with pytest.raises(ValueError, match='Expected classes'):
        predict_probabilities(model, features)


def test_report_and_matrix_order(tmp_path):
    actual = np.array([0, 0, 0, 1])
    predictions = np.array([0, 1, 1, 1])
    model = SimpleNamespace(predict=lambda frame: predictions)
    result = train.evaluate(model, pd.DataFrame({'x': actual}), actual, 'test', tmp_path)
    assert result['class_order'] == [0, 1]
    assert result['class_names'] == ['Higher truthfulness', 'Lower truthfulness']
    assert result['confusion_matrix'] == [[1, 2], [0, 1]]
    assert result['classification_report'][CLASS_NAMES[0]]['support'] == 3
    assert result['classification_report'][CLASS_NAMES[1]]['support'] == 1
    assert result['metrics']['binary_f1_lower_truthfulness'] == .5
    assert (tmp_path / 'confusion_matrix.png').is_file()
    assert json.loads((tmp_path / 'test_metrics.json').read_text()) == result


def test_serialization_and_relative_model_path(tmp_path, raw_claims, monkeypatch):
    features = prepare_features(raw_claims)
    model = train.build_pipeline(n_estimators=3).fit(features, raw_claims['label'].map(LABEL_MAPPING))
    save_model(model, tmp_path / 'artifacts' / 'model.joblib', {})
    monkeypatch.setattr(model_io, 'PROJECT_ROOT', tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    restored = load_model('artifacts/model.joblib')
    np.testing.assert_array_equal(model.predict(features), restored.predict(features))
    np.testing.assert_allclose(model.predict_proba(features), restored.predict_proba(features))
    with pytest.raises(FileNotFoundError, match='Model not found'):
        load_model('missing.joblib')
    joblib.dump(model, tmp_path / 'old.joblib')
    with pytest.raises(ValueError, match='Incompatible model format'):
        load_model(tmp_path / 'old.joblib')
    (tmp_path / 'corrupt.joblib').write_bytes(b'')
    with pytest.raises(ValueError, match='Cannot deserialize'):
        load_model(tmp_path / 'corrupt.joblib')


def test_end_to_end_train_only_fit_and_mlflow(tmp_path, raw_claims, write_tsv, monkeypatch):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    for split in ('train', 'valid', 'test'):
        frame = raw_claims.copy()
        frame['id'] = split + frame['id']
        if split != 'train':
            frame['statement'] = split + 'uniquetoken ' + frame['statement']
            frame[NUMERIC_FEATURES] = 100
        write_tsv(data_dir / f'{split}.tsv', frame)
    output_dir = tmp_path / 'artifacts'
    old_tracking_uri = train.mlflow.get_tracking_uri()
    try:
        model, results = train.train_model(data_dir, output_dir, dict.fromkeys(('train', 'valid', 'test'), 6), n_estimators=3)
        vocabulary = model['preprocessor'].named_transformers_['text'].vocabulary_
        assert 'validuniquetoken' not in vocabulary
        assert 'testuniquetoken' not in vocabulary
        np.testing.assert_allclose(model['preprocessor'].named_transformers_['num'].mean_, np.full(5, 17 / 6))
        test_data = load_and_process_data(data_dir / 'test.tsv')
        np.testing.assert_array_equal(
            confusion_matrix(test_data['target'], model.predict(test_data[FEATURE_COLUMNS]), labels=CLASS_ORDER),
            results['test']['confusion_matrix'],
        )
        restored = load_model(output_dir / 'model.joblib')
        np.testing.assert_array_equal(restored.predict(test_data[FEATURE_COLUMNS]), model.predict(test_data[FEATURE_COLUMNS]))
        metadata = json.loads((output_dir / 'training_metadata.json').read_text())
        assert metadata['datasets']['test']['sha256'] == hashlib.sha256((data_dir / 'test.tsv').read_bytes()).hexdigest()
        client = train.mlflow.MlflowClient()
        run = client.get_run(metadata['run_id'])
        assert run.info.status == 'FINISHED'
        assert run.data.metrics['test_accuracy'] == results['test']['metrics']['accuracy']
        assert run.data.params['rf_random_state'] == '42'
        assert 'confusion_matrix.png' in {Path(artifact.path).name for artifact in client.list_artifacts(run.info.run_id, 'evaluation')}
        tracked = train.mlflow.sklearn.load_model(metadata['mlflow_model_uri'])
        np.testing.assert_array_equal(tracked.predict(test_data[FEATURE_COLUMNS]), model.predict(test_data[FEATURE_COLUMNS]))
    finally:
        train.mlflow.set_tracking_uri(old_tracking_uri)


def test_split_overlap_rejected_before_training(tmp_path, raw_claims, write_tsv):
    for split in ('train', 'valid', 'test'):
        write_tsv(tmp_path / f'{split}.tsv', raw_claims)
    with pytest.raises(ValueError, match='Overlapping claim IDs'):
        train.train_model(tmp_path, tmp_path / 'output', dict.fromkeys(('train', 'valid', 'test'), 6), 3)
