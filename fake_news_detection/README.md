# Fake News Detection — LIAR Benchmark

Binary political-claim truthfulness classification on the LIAR dataset. This project does **not** independently verify news articles against external evidence.

## Current implementation

- `src/preprocess.py`: shared input schema, text normalization, contextual metadata, and binary target mapping.
- `src/train.py`: TF-IDF plus five historical truthfulness-count features and a Random Forest classifier, with MLflow logging.
- `src/app.py`: Streamlit interface for manual claims and test examples.

## Binary target mapping

| Original label | Target |
| --- | ---: |
| `pants-fire` | 1 (lower truthfulness) |
| `false` | 1 (lower truthfulness) |
| `barely-true` | 1 (lower truthfulness) |
| `half-true` | 0 (higher truthfulness group) |
| `mostly-true` | 0 (higher truthfulness group) |
| `true` | 0 (higher truthfulness group) |

This is a project-defined grouping of LIAR's six truthfulness ratings, not a universal definition of fake versus real news.

## Layout

The current training script expects `data/train.tsv`, `data/valid.tsv`, and `data/test.tsv` relative to the working directory. From the repository root, these files currently live in the root-level `data/` directory. Run commands from the repository root unless you update the data paths.

The `.dvc` pointer files under `fake_news_detection/data/` describe data outputs, but `fake_news_detection/.dvc/config` is currently empty. A configured DVC remote is required if DVC is used to retrieve the project-contained data.

## Setup

Use a virtual environment and install the project's dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r fake_news_detection/requirements.txt
```

The script uses relative data paths. If using the existing root-level files, run from the repository root:

```bash
python fake_news_detection/src/train.py
```

Start the Streamlit application from the repository root:

```bash
streamlit run fake_news_detection/src/app.py
```

Training writes an MLflow run and logs metrics/model artifacts to the configured MLflow tracking location. The app currently requires a run ID that is accessible from the same tracking store; a fresh environment will not have access to another machine's local tracking database or files.

## Evaluation and limitations

- Report accuracy, precision, recall, F1, and confusion matrix for each class.
- Choose models using validation data; reserve the test split for final evaluation.
- The app currently substitutes zero for all historical truthfulness-count features on manual input. Treat manual predictions as a different feature scenario until a text/metadata-only model is separately evaluated.
- `predict_proba` scores are not necessarily calibrated probabilities. Do not interpret them as verified factual confidence without calibration evaluation.
- Political-claim truthfulness is not the same task as checking the factual accuracy of a complete news article.
- No test metrics are published in this README until a fresh run reproduces them.

## Reproducibility

Before using this project as a benchmark:
1. Configure dataset paths and any DVC remote consistently.
2. Fix an explicit label mapping and validate unknown labels.
3. Align inference features and preprocessing with training.
4. Keep MLflow tracking databases and generated run files out of Git.
5. Record the Python and package versions used for the evaluated model.
