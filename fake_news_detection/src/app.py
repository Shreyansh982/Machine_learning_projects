from pathlib import Path

import pandas as pd
import streamlit as st

from model_io import load_model, predict_probabilities
from preprocess import (
    CLASS_NAMES, DATA_DIR, FEATURE_COLUMNS, HISTORY_COLUMNS, LABEL_MAPPING,
    NUMERIC_FEATURES, PROJECT_ROOT, TEXT_METADATA, load_and_process_data, prepare_features,
)

st.set_page_config(page_title='Political claim classifier', page_icon='🔎', layout='wide')
st.title('Political claim truthfulness estimates')
st.write('This model estimates a grouped LIAR label from claim text and speaker metadata. It does not verify facts or retrieve evidence.')
st.caption('; '.join(
    f"{target} = {name} ({', '.join(label for label, value in LABEL_MAPPING.items() if value == target)})"
    for target, name in CLASS_NAMES.items()
) + '. Probabilities are uncalibrated model outputs.')


@st.cache_resource
def cached_model(path, modified_ns):
    return load_model(path)


@st.cache_data
def cached_test_data(path, modified_ns):
    return load_and_process_data(path)


st.sidebar.header('Model')
model_path = st.sidebar.text_input('Model artifact path (relative to project)', value='artifacts/model.joblib')
try:
    resolved = Path(model_path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    model = cached_model(str(resolved), resolved.stat().st_mtime_ns)
except (OSError, ValueError, TypeError) as exc:
    st.sidebar.error(f'Cannot load model: {exc}. Train with: python fake_news_detection/src/train.py (repository root).')
    st.stop()
st.sidebar.success('Model loaded')
st.sidebar.caption(str(resolved))

if st.sidebar.button('Random test example'):
    try:
        path = DATA_DIR / 'test.tsv'
        test_data = cached_test_data(str(path), path.stat().st_mtime_ns)
        row = test_data.sample(1).iloc[0]
        for column in ['statement', *TEXT_METADATA, *NUMERIC_FEATURES]:
            value = row[column]
            st.session_state[column] = str(int(value)) if column in NUMERIC_FEATURES else ('' if pd.isna(value) else str(value))
        st.session_state['test_example'] = row.to_dict()
    except (OSError, ValueError) as exc:
        st.sidebar.error(f'Cannot load test examples: {exc}. Restore data/test.tsv as described in README.md.')

st.info('Supply all five known speaker history counts, excluding this claim. Leave unknown counts blank; prediction requires known counts. Enter 0 only for an actual zero. Optional text metadata can be left blank.')
with st.form('prediction_form'):
    columns = st.columns(4)
    for container, field, title in zip(columns, TEXT_METADATA, ['Party', 'Speaker', 'Subject', 'Context']):
        with container:
            st.text_input(title, key=field)
    st.text_area('Claim', key='statement')
    columns = st.columns(5)
    for container, (label, field) in zip(columns, HISTORY_COLUMNS.items()):
        with container:
            st.text_input(f'{label} count', key=field)
    submitted = st.form_submit_button('Estimate label')

if submitted:
    try:
        raw = pd.DataFrame([{field: st.session_state[field] for field in ['statement', *TEXT_METADATA, *NUMERIC_FEATURES]}])
        features = prepare_features(raw)
        example = st.session_state.get('test_example')
        if example is not None:
            original = prepare_features(pd.DataFrame([example])).reset_index(drop=True)
            if features.equals(original):
                label = example['label']
                st.info(f"Dataset label: {label} ({CLASS_NAMES[LABEL_MAPPING[label]]}). History counts exclude this claim.")
        probabilities = predict_probabilities(model, features)
        predicted = int(model.predict(features)[0])
        st.subheader(f'Model estimate: {CLASS_NAMES[predicted]}')
        for label, name in CLASS_NAMES.items():
            st.write(f'{name}: {probabilities[label][0]:.1%} model probability')
        st.caption('These probabilities are not calibrated confidence or evidence that the claim is true or false.')
        with st.expander('Prepared model inputs'):
            st.dataframe(features[FEATURE_COLUMNS])
    except (ValueError, TypeError) as exc:
        st.error(f'Cannot estimate this claim: {exc}')
