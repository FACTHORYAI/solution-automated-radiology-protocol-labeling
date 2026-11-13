"""One smoke test per job, on the synthetic sample data."""

from __future__ import annotations

from pathlib import Path

import pytest

from automated_radiology_protocol_labeling import io
from automated_radiology_protocol_labeling.config import Settings
from automated_radiology_protocol_labeling.jobs import REGISTRY, call_run
from automated_radiology_protocol_labeling.jobs.prepare_notes import Params as PrepareParams
from automated_radiology_protocol_labeling.schemas import (
    CONFUSION_MATRIX,
    EVALUATION_SUMMARY,
    LABEL_DATASET,
    LABEL_DISTRIBUTION,
    LABEL_METRICS,
    PREDICTIONS,
)
from automated_radiology_protocol_labeling.text.recorded import RecordedTextModel, load_answers

REPO = Path(__file__).resolve().parent.parent
SETTINGS = Settings(sample_dir=REPO / "data" / "sample", output_dir=REPO / "output")

SAMPLE_TABLES = (
    "notes_raw",
    "notes_test_raw",
    "label_dataset",
    "predictions",
    "evaluation_summary",
    "label_metrics",
    "confusion_matrix",
    "label_distribution",
)


@pytest.fixture(scope="module")
def recorded_model() -> RecordedTextModel:
    return RecordedTextModel(load_answers(io.read_json(REPO / "tests" / "recorded_answers.json")))


@pytest.fixture(scope="module")
def tables() -> dict[str, object]:
    return {name: io.read_table(name, SETTINGS) for name in SAMPLE_TABLES}


def run_job(name: str, tables: dict, params=None, model=None):
    spec = REGISTRY[name]
    return call_run(spec, tables, params if params is not None else spec.params_cls(), model)


def test_prepare_notes(tables: dict) -> None:
    outputs = run_job("prepare_notes", {"notes_raw": tables["notes_raw"]})
    dataset = outputs["label_dataset"]
    assert dataset.schema.equals(LABEL_DATASET)
    assert dataset.equals(tables["label_dataset"].select(LABEL_DATASET.names))
    assert dataset.num_rows == 80


def test_prepare_notes_drops_rare_labels(tables: dict) -> None:
    outputs = run_job(
        "prepare_notes", {"notes_raw": tables["notes_raw"]}, PrepareParams(min_label_count=100)
    )
    assert outputs["label_dataset"].num_rows == 0


def test_label_protocols(tables: dict, recorded_model: RecordedTextModel) -> None:
    outputs = run_job(
        "label_protocols",
        {
            "notes_test_raw": tables["notes_test_raw"],
            "label_dataset": tables["label_dataset"],
        },
        model=recorded_model,
    )
    predictions = outputs["predictions"]
    assert predictions.schema.equals(PREDICTIONS)
    assert predictions.equals(tables["predictions"].select(PREDICTIONS.names))
    assert predictions.num_rows == 20
    assert outputs["rejected_labels"].num_rows == 0


def test_label_protocols_rejects_unknown_label(tables: dict) -> None:
    class UnknownLabelModel:
        def complete(self, instructions: str, text: str) -> str:
            return '{"label": "Not A Protocol", "confidence": 0.9}'

    outputs = run_job(
        "label_protocols",
        {
            "notes_test_raw": tables["notes_test_raw"].slice(0, 1),
            "label_dataset": tables["label_dataset"],
        },
        model=UnknownLabelModel(),
    )
    assert outputs["predictions"].num_rows == 0
    rejected = outputs["rejected_labels"].to_pylist()
    assert len(rejected) == 1
    assert "not one of the known protocols" in rejected[0]["reason"]


def test_evaluate_predictions(tables: dict) -> None:
    outputs = run_job(
        "evaluate_predictions",
        {"predictions": tables["predictions"], "notes_test_raw": tables["notes_test_raw"]},
    )
    summary = outputs["evaluation_summary"]
    assert summary.schema.equals(EVALUATION_SUMMARY)
    assert summary.equals(tables["evaluation_summary"].select(EVALUATION_SUMMARY.names))
    row = summary.to_pylist()[0]
    assert row["notes"] == 20
    assert row["correct"] == 18
    assert row["accuracy"] == pytest.approx(0.9)

    metrics = outputs["label_metrics"]
    assert metrics.schema.equals(LABEL_METRICS)
    assert metrics.equals(tables["label_metrics"].select(LABEL_METRICS.names))
    rows = {row["label"]: row for row in metrics.to_pylist()}
    assert rows["Pulmonary Embolus"]["recall"] == pytest.approx(0.5)
    assert rows["CTA Aortic Dissection"]["precision"] == pytest.approx(2 / 3)

    matrix = outputs["confusion_matrix"]
    assert matrix.schema.equals(CONFUSION_MATRIX)
    misses = {
        (row["actual"], row["predicted"]): row["notes"]
        for row in matrix.to_pylist()
        if row["actual"] != row["predicted"]
    }
    assert misses[("Pulmonary Embolus", "CTA Aortic Dissection")] == 1
    assert misses[("CT Abdomen", "Renal Stone Protocol")] == 1


def test_label_summary(tables: dict) -> None:
    outputs = run_job("label_summary", {"label_dataset": tables["label_dataset"]})
    distribution = outputs["label_distribution"]
    assert distribution.schema.equals(LABEL_DISTRIBUTION)
    assert distribution.equals(tables["label_distribution"].select(LABEL_DISTRIBUTION.names))
    assert distribution.num_rows == 10
    assert {row["notes"] for row in distribution.to_pylist()} == {8}
