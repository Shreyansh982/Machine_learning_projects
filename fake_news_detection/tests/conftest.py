import csv
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from preprocess import COLUMNS, LABEL_MAPPING, NUMERIC_FEATURES


@pytest.fixture
def raw_claims():
    rows = []
    for index, label in enumerate(LABEL_MAPPING):
        row = dict.fromkeys(COLUMNS, '')
        row.update(id=f'{index}.json', label=label,
                   statement=f'A quoted "claim" about schools and taxes number {index}.',
                   subject='education', speaker='speaker', party_affiliation='party', context='debate')
        row.update(dict.fromkeys(NUMERIC_FEATURES, 3))
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def write_tsv():
    def write(path, frame):
        with Path(path).open('w', encoding='utf-8', newline='') as target:
            csv.writer(target, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar=None).writerows(frame[COLUMNS].itertuples(index=False, name=None))
        return path
    return write
