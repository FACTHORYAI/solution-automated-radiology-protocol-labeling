# Automated Radiology Protocol Labeling — agent instructions

You help members label radiology examination protocols from clinical notes, and judge how well the labelling performs.

## What the project provides

- Sample tables in `data/sample` (synthetic): `notes_raw` and `notes_test_raw` (labelled and held-out notes), `label_dataset` (the training set after deduplication and rare-label filtering), `predictions` (the protocol predicted per test note), and the evaluation tables `evaluation_summary`, `label_metrics`, `confusion_matrix`, `label_distribution`.
- Jobs under `src/automated_radiology_protocol_labeling/jobs`: `prepare_notes` (build the training set), `label_protocols` (predict a protocol per note through a `TextModel`), `evaluate_predictions` (accuracy, per-protocol metrics, confusion matrix), `label_summary` (label distribution).

## How to answer a question

1. Classify what the member asks: which protocol a note calls for (`label_protocols`), how accurate the labelling is (`evaluation_summary`), which protocols are confused with which (`confusion_matrix`, `label_metrics`), or how balanced the training set is (`label_distribution`).
2. Check the parameters with `uv run python -m automated_radiology_protocol_labeling <job> --help`; `min_label_count` drops protocols that are too rare to train on (the original pipeline used 50).
3. Run the job, or answer directly from the tables: `predictions` holds the predicted protocol and confidence per test note; `label_metrics` holds precision, recall, F1 and support per protocol; `confusion_matrix` holds the actual-versus-predicted counts.
4. Report results as a table with the counts behind them, and name the protocols that are confused. Never guess beyond the tables.

## Data care

- The sample data is synthetic; real patient notes never enter this project.
- `label_protocols` needs a model behind `text.interface`; the repository ships a recorded stand-in for tests and demos.
- A predicted protocol is a scheduling suggestion for a radiologist to review, not an order, and results are for research and analytics, not clinical decisions.
