# Political claim truthfulness classification

This project estimates a **grouped LIAR truthfulness label** using TF-IDF text, contextual metadata, five speaker history counts, and one Random Forest. It does not retrieve evidence or independently verify claims. It is a research demo for short political statements, not a reliable general news authenticity checker. There is no hyperparameter search, algorithm comparison, cross-validation, or probability calibration in this implementation.

## Data and label semantics

Source: William Yang Wang, [LIAR (ACL 2017)](https://aclanthology.org/P17-2067/), collected from PolitiFact. The author's [dataset archive](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) supplies `train.tsv`, `valid.tsv`, and `test.tsv`.

The files are UTF-8, tab-separated, **without a header**, with 14 fields in this order:

```text
id, label, statement, subject, speaker, job_title, state_info,
party_affiliation, barely_true_counts, false_counts, half_true_counts,
mostly_true_counts, pants_on_fire_counts, context
```

| Target | Display name | Source labels |
| --- | --- | --- |
| 0 | Higher truthfulness | half-true, mostly-true, true |
| 1 | Lower truthfulness | pants-fire, false, barely-true |

`src/preprocess.py` defines the mapping and class order `[0, 1]`. The binary positive class is explicitly `1`. Higher truthfulness includes partially true claims; neither class establishes a fact. Unknown or missing labels fail validation.

The expected split sizes are **10,269 / 1,284 / 1,283** rows. Loading checks row widths, row counts, required statements, label values, claim IDs, and numeric counts. Counts must be known, finite nonnegative integers; numeric strings are accepted. Missing counts are rejected. Optional text metadata becomes `none`; unused job title/state fields may be missing. Split IDs must not overlap.

### Speaker history and leakage correction

The source histories include the current claim, as explained in [the paper, section 2](https://aclanthology.org/P17-2067.pdf). The dataset loader subtracts one from the count matching the original six-way label, before training or evaluating. `true` has no count column. Existing zero counts stay at zero and produce a warning: the local files have 28 such inconsistencies in train, 1 in validation, and 5 in test. No rows are dropped or reassigned. Files on disk remain unchanged. The correction uses the known dataset label only to remove the current observation from source metadata; manual claims must supply history that already excludes the claim.

This removes direct self-count leakage, but the source histories are aggregate snapshots, not guaranteed historical data available at the time each claim was made. Speakers also recur across splits. These limits and political/source selection bias restrict how held-out results generalize to future or unfamiliar speakers.

## Installation and dataset setup

Use Python 3.11 or newer. Commands below use `python`; on Windows, `py` can be substituted. Run them from the **repository root**, `Machine_learning_projects`:

```powershell
python -m venv fake_news_detection/.venv
fake_news_detection/.venv/Scripts/Activate.ps1
python -m pip install -r fake_news_detection/requirements.txt
```

On Linux/macOS, activate with `source fake_news_detection/.venv/bin/activate` instead. Train and serve using the same environment; serialized scikit-learn models are version sensitive. The model bundle records the exact scikit-learn version, and loading rejects a mismatch. MLflow also records the model environment and signature.

Expected data location:

```text
fake_news_detection/data/train.tsv
fake_news_detection/data/valid.tsv
fake_news_detection/data/test.tsv
```

The repository already tracks identical source files under root `data/`. For a fresh checkout, copy them into the project directory:

```powershell
python -c "import shutil; from pathlib import Path; d=Path('fake_news_detection/data'); d.mkdir(parents=True, exist_ok=True); [shutil.copyfile(Path('data')/n, d/n) for n in ('train.tsv','valid.tsv','test.tsv')]"
```

Alternatively extract the original archive's three splits into `fake_news_detection/data/`. Training never downloads data automatically. `--data-dir` can select a directory containing these same original splits and sizes, including root `data/` directly.

### DVC status

The three `data/*.tsv.dvc` pointers match the local files:

| File | MD5 |
| --- | --- |
| train.tsv | cd568a4923572a9f97a8dd010a9c0477 |
| valid.tsv | b8910909172aa6d35b4539be4168022e |
| test.tsv | 2b72ac1cb806133c527858b745575364 |

`.dvc/config` is empty: **there is no remote configured**. A local DVC cache exists in this working copy, but Git does not distribute that cache. `dvc pull` cannot restore a fresh checkout from this configuration. Use the tracked root files or the author's archive; no credentials or remote storage have been invented. DVC is optional and is not a training dependency. The project copies are ignored by Git; the root source data remains tracked.

## Train and inspect evaluation

From repository root:

```powershell
python fake_news_detection/src/train.py
```

From the project directory the equivalent command is `python src/train.py`. From another directory, invoke the script by its absolute path. Default data, artifact, and tracking paths are derived from the script location, not the shell directory. Explicit `--data-dir` / `--output-dir` arguments follow normal shell-relative path rules.

The original split boundaries are preserved. TF-IDF (5,000 features, English stop words, unigrams/bigrams), numeric scaling, and the Random Forest fit **only the training split**. The forest uses 200 trees, balanced class weights, and seed 42. Its configuration is fixed in advance. Validation is reported without tuning; the same model then evaluates the held-out test set. The test set is never used for fitting or model selection.

Generated files are under `fake_news_detection/artifacts/`:

- `model.joblib`: complete compressed model bundle used by the app, with feature/label/version metadata.
- `confusion_matrix.png`: newly computed test-set matrix, ordered Higher truthfulness (0), Lower truthfulness (1); rows are actual classes and columns are predictions.
- `valid_metrics.json`, `test_metrics.json`: class order, classification report, confusion matrix counts, and scalar metrics.
- `valid_classification_report.txt`, `test_classification_report.txt`: readable precision, recall, F1, support, macro/weighted averages, and accuracy.
- `training_metadata.json`: split sizes, SHA-256 hashes, missing-value counts, history inconsistencies, run ID and MLflow model URI.
- `tracking/mlflow.db`, `tracking/mlruns/`: local experiment store and complete MLflow model/artifacts.

Binary F1 explicitly uses lower truthfulness as positive. Per-class precision, recall, F1, support, macro F1, weighted F1 and accuracy are logged for validation and test. MLflow receives the forest/text parameters, seed, confusion matrix, reports, provenance, and a serialized scikit-learn model. Outputs are regenerated on each invocation; the local bundle represents the last completed training call, and each MLflow run preserves its own artifacts.

Read the generated report:

```powershell
python -c "from pathlib import Path; print(Path('fake_news_detection/artifacts/test_classification_report.txt').read_text())"
```

Start MLflow UI from repository root:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///fake_news_detection/artifacts/tracking/mlflow.db
```

Select experiment `LIAR_Truthfulness_RF`. The new store is separate from legacy `mlflow.db` / `mlruns/`; no direct SQL rewriting of old experiment locations occurs. Fresh training creates tracking locations for the current machine. When moving a checkout, regenerate tracking outputs; the app's local model bundle can be moved with its matching Python environment and needs no MLflow database.

## Launch the app

```powershell
python -m streamlit run fake_news_detection/src/app.py
```

The sidebar explicitly selects `artifacts/model.joblib`, resolved relative to `fake_news_detection/`. It also accepts an absolute path to a compatible bundle generated by this training script. It never auto-selects a run, relies on hardcoded IDs, or searches old databases. Modified artifacts invalidate the loading cache. Missing, corrupt, old-format, or incompatible models show a training/reinstallation instruction.

Speaker, party, subject, and context are optional. **All five history counts are required** for a manual prediction; blank means unknown and prevents prediction. Enter zero only for a known zero. This avoids inventing a statistically unjustified history for an unknown speaker. There is no text-only fallback or imputation policy. Model probabilities are explicitly uncalibrated, not verified facts or confidence estimates. Class probabilities are associated via `model.classes_`, and the displayed prediction uses `model.predict`.

`Random test example` uses the same dataset loader, count correction, text preparation, numeric types, and label mapping as evaluation. The source label is shown on submission only while the prepared inputs still match the selected example. Changing a claim or its metadata removes that association. Missing test data produces an actionable message while manual inference remains available.

## Legacy artifacts and Git hygiene

Git tracks legacy `MLmodel` and environment metadata without the referenced `model.pkl` binaries. This working copy also contains 24 ignored legacy binaries under the project's `mlruns/`, and a sampled local model was successfully loaded. Thus local models exist, but the Git metadata alone does not form a usable artifact in a fresh checkout. Their absolute artifact paths include earlier locations under `C:/Users/SHREYANSH RATHORE/OneDrive/Desktop/` and `.../Machine_learning_projects/`. No other user's path was found in the inspected metadata. Regenerate a compatible portable bundle with the train command above. The old project-level `confusion_matrix.png` is not the new evaluation output.

Existing experiment history, models, databases, and old images have been preserved. Ignore rules prevent newly generated tracking, caches, logs, materialized data, and model outputs from being added. **Ignoring does not untrack existing files**: legacy tracked database/image entries and pre-existing staged changes still need a deliberate repository cleanup if desired. This task does not alter the Git index, commit, push, or rewrite history. `tempCodeRunnerFile.python` is a preserved legacy scratch file and is not part of the supported workflow.

## Offline verification

```powershell
python -m pytest fake_news_detection/tests -q
python -m compileall -q fake_news_detection/src fake_news_detection/tests
git diff --check -- fake_news_detection
```

Tests use temporary synthetic datasets and local MLflow storage, without APIs, downloads, or DVC remotes. They cover invalid labels/schema/counts, label correction, feature consistency, explicit class order, reversed probability ordering, working-directory independence, model serialization, split isolation, MLflow artifacts, and Streamlit interactions. Real-data metrics are generated only by running the training command; the README intentionally does not promise a particular accuracy.
