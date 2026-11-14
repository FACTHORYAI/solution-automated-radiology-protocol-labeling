<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FACTHORYAI/.github/main/brand/word-mark-only-mono.svg">
    <img alt="Facthory" src="https://raw.githubusercontent.com/FACTHORYAI/.github/main/brand/word-mark-only.svg" width="360">
  </picture>
</p>

# Automated Radiology Protocol Labeling

Label radiology examination protocols from clinical notes automatically, in Facthory.

![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue) ![Runs on Facthory](https://img.shields.io/badge/runs%20on-Facthory-1f6feb) ![Status: install coming](https://img.shields.io/badge/status-install%20coming-lightgrey)

> **Install coming.** This is a Facthory solution: its data, analysis code, agent skill and starting pages load into a Facthory workspace as a ready project. Installing from this page is not available yet; this repository is being prepared for it.

## What it does

Automated Radiology Protocol Labeling reads the free text of a clinical note, such as "58M with sudden tearing chest pain radiating to the back. History: Prior aortic root repair.", and suggests which radiology examination protocol it calls for: CTA Aortic Dissection, CT Urography, Renal Stone Protocol, C-Spine(Bone), L-Spine, Routine Sinus, Pulmonary Embolus, CTA Brain, CT Abdomen or Brain Perfusion. It then measures how good the suggestions are, as tables.

1. **Prepare.** `prepare_notes` takes the labelled notes, drops repeated texts, and drops protocols that occur fewer than `min_label_count` times. What remains is the training set and the list of protocols a prediction may use.
2. **Label.** `label_protocols` reads each held-out test note through a text model, which chooses one protocol from that list and gives a confidence between 0 and 1. An answer that is not JSON, names a protocol outside the list or has no usable confidence goes to `rejected_labels` with the reason, and never becomes a prediction.
3. **Evaluate.** `evaluate_predictions` compares the predictions with the ground truth of the test notes and returns three tables: the accuracy, precision, recall and F1 per protocol, and the confusion matrix (actual against predicted protocol, with note counts).
4. **Summarise.** `label_summary` counts the training notes per protocol, to show how balanced the training set is.

Step 2 needs a text model. Steps 1, 3 and 4 are plain SQL on DuckDB and need none. A suggested protocol is for a radiologist to review, not an order.

## How it works in Facthory

- Jobs read tables and write tables. They draw no screens and own no storage; querying, charting, joining with your own data, permissions and audit belong to the platform.
- The agent skill in `skills/` and the starter page in `pages/` describe how to work with the tables and the jobs.
- Sample data in this repository is synthetic. No real patient data.

## What the result could look like

![Mock-up of a Facthory report page titled "Protocol labelling results", marked "Sample data": four figures (notes labelled, accuracy, protocols covered, predictions below 0.8 confidence), a table of the lowest-confidence predictions with the suggested protocol and a Correct or Check chip, a confusion matrix of the ten protocols, and a bar chart of the F1 score per protocol.](docs/images/protocol-labelling-mockup.png)

A design illustration of a report built from this solution's tables, drawn from this repository's own sample data (invented notes and invented predictions). It is a mock-up and not a screenshot of a released feature; the accuracy shown compares invented predictions with the ground truth.

## Background: adapting a model by fine-tuning

![Two diagrams side by side. LoRA: the inputs go through two small trained matrices Wa and Wb and, in parallel, through frozen pretrained weights kept in 16-bit floating point; the two paths are added to give the embedding, h = Wx + ΔWx. QLoRA: the same, with the frozen pretrained weights quantized to 4-bit integers.](docs/images/lora-and-qlora.png)

A way to adapt a large model to labelled notes is to fine-tune it with LoRA, which trains only two small matrices beside the frozen pretrained weights, or with QLoRA, which also stores the frozen weights as 4-bit integers to save memory. This is background only: this repository does not fine-tune a model. It asks a model, through the interface in `text/`, to choose a protocol from the known list, and it does not train on the labelled notes.

## Jobs

| Job | What it does | Inputs | Outputs |
| --- | --- | --- | --- |
| `prepare_notes` | Builds the training set: deduplicates notes and drops protocols too rare to learn | `notes_raw` | `label_dataset` |
| `label_protocols` | Predicts the radiology protocol each test note calls for, with confidence | `notes_test_raw`, `label_dataset` | `predictions`, `rejected_labels` |
| `evaluate_predictions` | Accuracy, per-protocol precision/recall/F1 and the confusion matrix | `predictions`, `notes_test_raw` | `evaluation_summary`, `label_metrics`, `confusion_matrix` |
| `label_summary` | Notes per protocol in the training set | `label_dataset` | `label_distribution` |

`label_protocols` takes a `TextModel` (see `src/automated_radiology_protocol_labeling/text/interface.py`) as a third argument; Facthory provides model access, and a prediction is validated against the known protocol vocabulary before it becomes a row — an unknown protocol goes to `rejected_labels` with the reason. Named measures live in `metrics/`.

## Data

The sample tables in `data/sample/` are synthetic and CC0-1.0: `notes_raw`, `notes_test_raw`, `label_dataset`, `predictions`, `evaluation_summary`, `label_metrics`, `confusion_matrix` and `label_distribution`. They hold 200 invented labelled notes and 100 invented held-out notes over ten protocols (20 and 10 per protocol), 100 predictions, an accuracy of 92 percent and a 17-cell confusion matrix, made by `data/generate_sample.py`. See [data/README.md](data/README.md) for origin and licence per file. No real patient data.

## Run it locally

```
uv sync
uv run python -m automated_radiology_protocol_labeling prepare_notes --min-label-count 2
uv run python -m automated_radiology_protocol_labeling label_protocols --model-answers tests/recorded_answers.json
uv run python -m automated_radiology_protocol_labeling evaluate_predictions
uv run pytest
```

`uv run python -m automated_radiology_protocol_labeling --help` lists every job; each job has its own `--help`. `label_protocols` additionally takes `--model-answers <file>` with a recorded-answers JSON (see `tests/recorded_answers.json`), since it reads notes through a model behind `text.interface`; the other jobs run without any model. Verified on the synthetic sample data only, not yet in a Facthory workspace.

A predicted protocol is a scheduling suggestion for a radiologist to review, not an order: `label_metrics` shows precision, recall and F1 per protocol, and `confusion_matrix` shows which protocols get mixed up.

## Open points

- **No text model ships.** `label_protocols` runs through the `TextModel` interface. The only implementation is `text/recorded.py`, which replays recorded answers for the sample test notes. A model must be plugged in behind `text/interface.py` before new notes can be labelled.
- **The labelled notes are not used for training or as examples today.** `label_protocols` takes only the protocol names from `label_dataset` and asks a model zero-shot to choose one. A classifier trained on the labelled notes (for example TF-IDF with logistic regression in scikit-learn), or similar labelled notes retrieved and shown to the model as examples, is the planned next step. Either would be compared with the model class by the evaluation tables.
- **Fine-tuning is not included.** The background section describes LoRA and QLoRA; this repository does not train or adapt any model.
- **The predictions in `predictions` are invented** to exercise the metrics: they equal the ground truth except for 8 of the 100 test notes, which get a protocol they are plausibly mixed up with, with lower confidence. They are not the output of a model, and the accuracy, per-protocol metrics and confusion matrix say nothing about any labelling method.
- **A protocol suggestion is for a radiologist to review.** It is not an order and not a clinical decision.
- **`min_label_count` defaults to 2**, which suits the sample (20 notes per protocol). A real training set needs a higher value to leave out protocols too rare to learn from.
- **The sample is invented.** It is generated from templates, so its notes are cleaner and shorter than real clinical notes, and it shows the shape of the tables, not realistic volumes or accuracy. The ten protocols and their wording are a small fixed list.
- **Installing into Facthory is not available yet.** The code is verified on the sample data on a laptop only.
- **The Dockerfile has not been built or verified.**
- **DuckDB 2.x only.** The code uses the DuckDB 2.0 Arrow API (`to_arrow_table`).
- No orchestrator or DAG files: Facthory, or a scheduler, calls the jobs.

## Not for clinical use

This is software for research and analytics. It is not a medical device and must not be used to diagnose or treat a patient. Results can be wrong; a qualified person reviews them.

## Disclaimers

Bandpey GmbH (“Facthory”) does not dispense medical, diagnosis, or treatment advice. This Solution (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Facthory to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Facthory a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this tool within a HIPAA Account.

## Licence

Apache-2.0. See [LICENSE](LICENSE). "Facthory" and the Facthory logo are trademarks of their owner and are not covered by that licence. This includes the logo shown in the image under `docs/images/`.

