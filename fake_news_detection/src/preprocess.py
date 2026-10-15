import csv
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
ARTIFACT_DIR = PROJECT_ROOT / 'artifacts'
COLUMNS = [
    'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
    'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts',
    'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context',
]
LABEL_MAPPING = {
    'pants-fire': 1, 'false': 1, 'barely-true': 1,
    'half-true': 0, 'mostly-true': 0, 'true': 0,
}
CLASS_NAMES = {0: 'Higher truthfulness', 1: 'Lower truthfulness'}
CLASS_ORDER = list(CLASS_NAMES)
POSITIVE_CLASS = 1
HISTORY_COLUMNS = {
    'barely-true': 'barely_true_counts', 'false': 'false_counts',
    'half-true': 'half_true_counts', 'mostly-true': 'mostly_true_counts',
    'pants-fire': 'pants_on_fire_counts',
}
NUMERIC_FEATURES = list(HISTORY_COLUMNS.values())
TEXT_METADATA = ['party_affiliation', 'speaker', 'subject', 'context']
FEATURE_COLUMNS = ['combined_text', *NUMERIC_FEATURES]
FEATURE_VERSION = 'liar-excluding-current-v1'


def clean_text(text):
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower())


def prepare_features(frame):
    """Prepare raw claims with known history counts that exclude the current claim."""
    required = ['statement', *TEXT_METADATA, *NUMERIC_FEATURES]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing feature columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('No claims provided.')
    statements = frame['statement']
    if statements.isna().any() or not statements.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError('Every claim needs a nonempty text statement.')
    features = pd.DataFrame(index=frame.index)
    text = frame[TEXT_METADATA].fillna('none').replace(r'^\s*$', 'none', regex=True)
    features['combined_text'] = text.astype(str).agg(' '.join, axis=1) + ' ' + statements.map(clean_text)
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[column], errors='coerce')
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(
                f'{column} must contain known, finite, nonnegative integer counts. '
                'Supply all five history counts; unknown is not zero.'
            )
        features[column] = values.astype('float64')
    return features[FEATURE_COLUMNS]


def load_and_process_data(filepath, expected_rows=None):
    path = Path(filepath)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f'Dataset not found: {path}. See README.md for dataset setup.')
    # LIAR quotes are literal text, not CSV quoting. Validate rows before naming columns.
    with path.open(encoding='utf-8', newline='') as source:
        rows = list(csv.reader(source, delimiter='\t', quoting=csv.QUOTE_NONE))
    if not rows or any(len(row) != len(COLUMNS) for row in rows):
        raise ValueError(f'{path}: expected nonempty, headerless TSV rows with {len(COLUMNS)} columns.')
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f'{path}: expected {expected_rows} rows, found {len(rows)}.')
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.mask(df.eq(''))
    if df['id'].isna().any() or df['id'].duplicated().any():
        raise ValueError(f'{path}: missing or duplicate claim IDs.')
    if not df['label'].isin(LABEL_MAPPING).all():
        invalid = df.loc[~df['label'].isin(LABEL_MAPPING), 'label'].unique().tolist()
        raise ValueError(f'{path}: invalid or missing labels: {invalid}')
    df['target'] = df['label'].map(LABEL_MAPPING).astype('int64')
    validated = prepare_features(df)
    df[NUMERIC_FEATURES] = validated[NUMERIC_FEATURES]
    inconsistent = 0
    for label, column in HISTORY_COLUMNS.items():
        mask = df['label'].eq(label)
        inconsistent += int((mask & df[column].eq(0)).sum())
        # Source histories include this claim; preserve anomalous existing zeros.
        df.loc[mask, column] = (df.loc[mask, column] - 1).clip(lower=0)
    if inconsistent:
        warnings.warn(
            f'{path.name}: {inconsistent} current-label history counts were already zero; kept at zero.',
            UserWarning,
        )
    df['combined_text'] = validated['combined_text']
    df.attrs['history_zero_inconsistencies'] = inconsistent
    return df
