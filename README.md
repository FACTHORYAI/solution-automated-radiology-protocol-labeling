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

Label radiology examination protocols from clinical notes automatically, in Facthory.

## How it will work in Facthory

- You install it into a workspace and get a project: sample data, the analysis, the agent's instructions and starting pages.
- Its results are tables and pages in your workspace, so you can query them, chart them and ask the agent about them, inside your permissions.
- Sample data in this repository is synthetic or openly licensed. No real patient data.

## Background: adapting a model by fine-tuning

![Two diagrams side by side. LoRA: the inputs go through two small trained matrices Wa and Wb and, in parallel, through frozen pretrained weights kept in 16-bit floating point; the two paths are added to give the embedding, h = Wx + ΔWx. QLoRA: the same, with the frozen pretrained weights quantized to 4-bit integers.](docs/images/lora-and-qlora.png)

A way to adapt a large model to labelled notes is to fine-tune it with LoRA, which trains only two small matrices beside the frozen pretrained weights, or with QLoRA, which also stores the frozen weights as 4-bit integers to save memory. This is background only: this repository does not fine-tune a model. It asks a model, through the interface in `text/`, to choose a protocol from the known list, and it does not train on the labelled notes.

## Not for clinical use

This is software for research and analytics. It is not a medical device and must not be used to diagnose or treat a patient. Results can be wrong; a qualified person reviews them.

## Licence

Apache-2.0. See [LICENSE](LICENSE). "Facthory" and the Facthory logo are trademarks of their owner and are not covered by that licence. This includes the logo shown in the image under `docs/images/`.

## Jobs

| Job | What it does | Inputs | Outputs |
| --- | --- | --- | --- |
| `prepare_notes` | Builds the training set: deduplicates notes and drops protocols too rare to learn | `notes_raw` | `label_dataset` |
| `label_protocols` | Predicts the radiology protocol each test note calls for, with confidence | `notes_test_raw`, `label_dataset` | `predictions`, `rejected_labels` |
| `evaluate_predictions` | Accuracy, per-protocol precision/recall/F1 and the confusion matrix | `predictions`, `notes_test_raw` | `evaluation_summary`, `label_metrics`, `confusion_matrix` |
| `label_summary` | Notes per protocol in the training set | `label_dataset` | `label_distribution` |

`label_protocols` takes a `TextModel` (see `src/automated_radiology_protocol_labeling/text/interface.py`) as a third argument; Facthory provides model access, and a prediction is validated against the known protocol vocabulary before it becomes a row — an unknown protocol goes to `rejected_labels` with the reason. Named measures live in `metrics/`.

## Data

The sample tables in `data/sample/` are synthetic and openly licensed (CC0-1.0): `notes_raw`, `notes_test_raw`, `label_dataset`, `predictions`, `evaluation_summary`, `label_metrics`, `confusion_matrix` and `label_distribution`. See [data/README.md](data/README.md) for one row per file with origin and licence. No real patient data.

## Run it locally

```
uv sync
uv run python -m automated_radiology_protocol_labeling prepare_notes --min-label-count 2
uv run python -m automated_radiology_protocol_labeling label_protocols --model-answers tests/recorded_answers.json
uv run python -m automated_radiology_protocol_labeling evaluate_predictions
uv run pytest
```

`uv run python -m automated_radiology_protocol_labeling --help` lists every job; each job has its own `--help`. `label_protocols` additionally takes `--model-answers <file>` with a recorded-answers JSON (see `tests/recorded_answers.json`), since it reads notes through a model behind `text.interface`; the other jobs run without any model.

A predicted protocol is a scheduling suggestion for a radiologist to review, not an order: `label_metrics` shows precision, recall and F1 per protocol, and `confusion_matrix` shows which protocols get mixed up.

