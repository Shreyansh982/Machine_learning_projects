from pathlib import Path
import pickle

import joblib
import numpy as np
import sklearn

from preprocess import CLASS_ORDER, FEATURE_COLUMNS, FEATURE_VERSION, LABEL_MAPPING, PROJECT_ROOT


def save_model(model, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'feature_version': FEATURE_VERSION,
                 'label_mapping': LABEL_MAPPING, 'sklearn_version': sklearn.__version__,
                 'metadata': metadata}, path, compress=3)


def load_model(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f'Model not found: {path}. Run: python fake_news_detection/src/train.py (from repository root).')
    try:
        bundle = joblib.load(path)
    except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError, ImportError, AttributeError, KeyError) as exc:
        raise ValueError(f'Cannot deserialize {path}: {exc}. Reinstall requirements and retrain.') from exc
    if not isinstance(bundle, dict) or bundle.get('feature_version') != FEATURE_VERSION or bundle.get('label_mapping') != LABEL_MAPPING:
        raise ValueError('Incompatible model format or label mapping. Retrain with src/train.py.')
    if bundle.get('sklearn_version') != sklearn.__version__:
        raise ValueError('Model scikit-learn version differs from this environment. Reinstall the training version or retrain.')
    model = bundle.get('model')
    if (not hasattr(model, 'predict_proba') or set(getattr(model, 'classes_', [])) != set(CLASS_ORDER)
            or list(getattr(model, 'feature_names_in_', [])) != FEATURE_COLUMNS):
        raise ValueError('Model classes or feature schema are incompatible. Retrain with src/train.py.')
    return model


def predict_probabilities(model, features):
    classes = list(model.classes_)
    if len(classes) != len(CLASS_ORDER) or set(classes) != set(CLASS_ORDER):
        raise ValueError(f'Expected classes {CLASS_ORDER}, found {classes}.')
    probabilities = np.asarray(model.predict_proba(features))
    if probabilities.shape != (len(features), len(classes)):
        raise ValueError('Model returned an incompatible probability shape.')
    return {label: probabilities[:, classes.index(label)] for label in CLASS_ORDER}
