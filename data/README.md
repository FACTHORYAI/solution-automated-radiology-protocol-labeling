# Sample data

Every file below is synthetic, generated for this repository from hand-written note
templates; no real patient data. Licence: CC0-1.0. The folder is a few kilobytes in
total, far under the 5 MB per file and 20 MB per folder limits.

| File | What it is | Where it came from | Licence | Synthetic |
| --- | --- | --- | --- | --- |
| `notes_raw.parquet` | 80 labelled clinical notes: note_id, text, label | Hand-written templates for this repository | CC0-1.0 | yes |
| `notes_test_raw.parquet` | 20 held-out notes with their protocol: note_id, text, ground_truth | Hand-written templates for this repository | CC0-1.0 | yes |
| `label_dataset.parquet` | The distinct notes whose protocol occurs often enough to train on | `prepare_notes` | CC0-1.0 | yes |
| `predictions.parquet` | The protocol the model predicted per test note, with confidence | `label_protocols` | CC0-1.0 | yes |
| `evaluation_summary.parquet` | Accuracy, correct and total over the test notes | `evaluate_predictions` | CC0-1.0 | yes |
| `label_metrics.parquet` | Precision, recall and F1 per protocol | `evaluate_predictions` | CC0-1.0 | yes |
| `confusion_matrix.parquet` | Actual against predicted protocol, with note counts | `evaluate_predictions` | CC0-1.0 | yes |
| `label_distribution.parquet` | Notes per protocol in the training set | `label_summary` | CC0-1.0 | yes |

Ten radiology protocols are covered (CTA Aortic Dissection, CT Urography, Renal Stone
Protocol, C-Spine(Bone), L-Spine, Routine Sinus, Pulmonary Embolus, CTA Brain,
CT Abdomen, Brain Perfusion) with eight training notes and two test notes each.

The recorded predictions in `predictions.parquet` deliberately contain two errors, so
the evaluation tables have signal: a pulmonary embolus note predicted as an aortic
dissection, and an abdominal pain note predicted as a renal stone protocol.

The original solution's dataset was **not** republished: the source repository is
distributed under the Databricks proprietary licence, which does not permit reuse
outside the Databricks platform. This repository ships its own synthetic notes instead.
