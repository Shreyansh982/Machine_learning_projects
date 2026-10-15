from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import model_io
import preprocess
from model_io import save_model
from preprocess import NUMERIC_FEATURES, prepare_features
from train import build_pipeline

APP_PATH = Path(__file__).resolve().parents[1] / 'src' / 'app.py'


@pytest.fixture
def app(tmp_path, raw_claims, write_tsv, monkeypatch):
    features = prepare_features(raw_claims)
    model = build_pipeline(n_estimators=3).fit(features, raw_claims['label'].map(preprocess.LABEL_MAPPING))
    save_model(model, tmp_path / 'artifacts' / 'model.joblib', {})
    (tmp_path / 'data').mkdir()
    write_tsv(tmp_path / 'data' / 'test.tsv', raw_claims)
    monkeypatch.setattr(preprocess, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(preprocess, 'DATA_DIR', tmp_path / 'data')
    monkeypatch.setattr(model_io, 'PROJECT_ROOT', tmp_path)
    return AppTest.from_file(APP_PATH, default_timeout=30).run()


def test_app_requires_known_counts_then_predicts(app):
    assert not app.exception
    assert app.text_input(key='barely_true_counts').value == ''
    app.text_area(key='statement').set_value('A claim about education.')
    app.main.button[0].click().run()
    assert 'unknown is not zero' in app.error[0].value
    for column in NUMERIC_FEATURES:
        app.text_input(key=column).set_value('0')
    app.main.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert 'Model estimate:' in app.subheader[0].value
    assert any('not calibrated confidence' in caption.value for caption in app.caption)


def test_app_example_uses_original_features_and_clears_stale_label(app):
    app.sidebar.button[0].click().run()
    assert not app.exception
    example = app.session_state['test_example']
    for column in NUMERIC_FEATURES:
        assert float(app.text_input(key=column).value) == example[column]
    app.main.button[0].click().run()
    assert not app.exception
    assert any('Dataset label:' in message.value for message in app.info)
    app.text_area(key='statement').set_value('An edited unrelated claim.')
    app.main.button[0].click().run()
    assert not any('Dataset label:' in message.value for message in app.info)


def test_app_missing_model_is_actionable(app):
    app.sidebar.text_input[0].set_value('artifacts/missing.joblib').run()
    assert not app.exception
    assert 'Cannot load model' in app.sidebar.error[0].value
    assert 'train.py' in app.sidebar.error[0].value


def test_app_missing_examples_are_actionable(app, monkeypatch, tmp_path):
    monkeypatch.setattr(preprocess, 'DATA_DIR', tmp_path / 'absent')
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert 'Cannot load test examples' in app.sidebar.error[0].value
