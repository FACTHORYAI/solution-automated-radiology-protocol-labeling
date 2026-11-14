"""One smoke test per job, on the synthetic sample data."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import duckdb
import pyarrow as pa
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
    NOTES_RAW,
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


def rows(tables: dict, name: str) -> list[dict]:
    return tables[name].to_pylist()


def test_sample_is_a_mix_of_notes(tables: dict) -> None:
    train = rows(tables, "notes_raw")
    test = rows(tables, "notes_test_raw")
    assert len(train) == 200
    assert len(test) == 100
    texts = [row["text"] for row in train + test]
    assert len(set(texts)) == len(texts)
    assert len({row["note_id"] for row in train + test}) == len(texts)
    protocols = {row["label"] for row in train}
    assert len(protocols) == 10
    assert {row["ground_truth"] for row in test} == protocols
    assert set(Counter(row["label"] for row in train).values()) == {20}
    assert set(Counter(row["ground_truth"] for row in test).values()) == {10}


def test_recorded_answers_match_the_prompt_and_the_test_notes(tables: dict) -> None:
    answers = io.read_json(REPO / "tests" / "recorded_answers.json")
    tasks = {
        io.read_prompt(spec.name).splitlines()[0].removeprefix("task:").strip()
        for spec in REGISTRY.values()
        if spec.needs_model
    }
    assert set(answers) == tasks
    texts = {row["text"] for row in rows(tables, "notes_test_raw")}
    assert all(set(task_answers) == texts for task_answers in answers.values())


def test_prepare_notes(tables: dict) -> None:
    outputs = run_job("prepare_notes", {"notes_raw": tables["notes_raw"]})
    dataset = outputs["label_dataset"]
    assert dataset.schema.equals(LABEL_DATASET)
    assert dataset.num_rows > 0
    assert dataset.equals(tables["label_dataset"].select(LABEL_DATASET.names))
    # Every raw note is distinct and every protocol is common enough, so none is dropped.
    expected = {(row["note_id"], row["text"], row["label"]) for row in rows(tables, "notes_raw")}
    assert {(r["note_id"], r["text"], r["label"]) for r in dataset.to_pylist()} == expected


def test_prepare_notes_drops_rare_labels_and_repeated_notes(tables: dict) -> None:
    outputs = run_job(
        "prepare_notes", {"notes_raw": tables["notes_raw"]}, PrepareParams(min_label_count=100)
    )
    assert outputs["label_dataset"].num_rows == 0

    notes = [
        {"note_id": "a", "text": "Notes: one.", "label": "L-Spine"},
        {"note_id": "b", "text": "Notes: one.", "label": "L-Spine"},
        {"note_id": "c", "text": "Notes: two.", "label": "L-Spine"},
        {"note_id": "d", "text": "Notes: three.", "label": "Rare"},
    ]
    outputs = run_job(
        "prepare_notes",
        {"notes_raw": pa.Table.from_pylist(notes, schema=NOTES_RAW)},
        PrepareParams(min_label_count=2),
    )
    assert [(row["note_id"], row["label"]) for row in outputs["label_dataset"].to_pylist()] == [
        ("a", "L-Spine"),
        ("c", "L-Spine"),
    ]


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
    assert predictions.num_rows == tables["notes_test_raw"].num_rows
    assert predictions.equals(tables["predictions"].select(PREDICTIONS.names))
    assert outputs["rejected_labels"].num_rows == 0
    # Each prediction is the recorded answer for its note.
    answers = io.read_json(REPO / "tests" / "recorded_answers.json")["label_protocols"]
    texts = {row["note_id"]: row["text"] for row in rows(tables, "notes_test_raw")}
    for row in predictions.to_pylist():
        assert json.loads(answers[texts[row["note_id"]]]) == {
            "label": row["predicted_label"],
            "confidence": row["confidence"],
        }


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
    assert rejected[0]["note_id"] == rows(tables, "notes_test_raw")[0]["note_id"]
    assert "not one of the known protocols" in rejected[0]["reason"]
    assert "Not A Protocol" in rejected[0]["model_answer"]


def test_evaluate_predictions(tables: dict) -> None:
    outputs = run_job(
        "evaluate_predictions",
        {"predictions": tables["predictions"], "notes_test_raw": tables["notes_test_raw"]},
    )
    truth = {row["note_id"]: row["ground_truth"] for row in rows(tables, "notes_test_raw")}
    pairs = [(truth[row["note_id"]], row["predicted_label"]) for row in rows(tables, "predictions")]
    assert pairs
    correct = sum(actual == predicted for actual, predicted in pairs)
    assert 0 < correct < len(pairs)

    summary = outputs["evaluation_summary"]
    assert summary.schema.equals(EVALUATION_SUMMARY)
    assert summary.num_rows == 1
    assert summary.equals(tables["evaluation_summary"].select(EVALUATION_SUMMARY.names))
    row = summary.to_pylist()[0]
    assert (row["notes"], row["correct"]) == (len(pairs), correct)
    assert row["accuracy"] == pytest.approx(correct / len(pairs))

    metrics = outputs["label_metrics"]
    assert metrics.schema.equals(LABEL_METRICS)
    assert metrics.num_rows > 0
    assert metrics.equals(tables["label_metrics"].select(LABEL_METRICS.names))
    actual_counts = Counter(actual for actual, _ in pairs)
    predicted_counts = Counter(predicted for _, predicted in pairs)
    right = Counter(actual for actual, predicted in pairs if actual == predicted)
    by_label = {r["label"]: r for r in metrics.to_pylist()}
    assert set(by_label) == set(actual_counts) | set(predicted_counts)
    for label, got in by_label.items():
        precision = right[label] / predicted_counts[label] if predicted_counts[label] else 0.0
        recall = right[label] / actual_counts[label] if actual_counts[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        assert (got["true_positive"], got["support"], got["predicted"]) == (
            right[label],
            actual_counts[label],
            predicted_counts[label],
        )
        assert got["precision"] == pytest.approx(precision)
        assert got["recall"] == pytest.approx(recall)
        assert got["f1"] == pytest.approx(f1)

    matrix = outputs["confusion_matrix"]
    assert matrix.schema.equals(CONFUSION_MATRIX)
    assert matrix.num_rows > 0
    assert matrix.equals(tables["confusion_matrix"].select(CONFUSION_MATRIX.names))
    assert {(r["actual"], r["predicted"]): r["notes"] for r in matrix.to_pylist()} == dict(
        Counter(pairs)
    )


def test_label_summary(tables: dict) -> None:
    outputs = run_job("label_summary", {"label_dataset": tables["label_dataset"]})
    distribution = outputs["label_distribution"]
    assert distribution.schema.equals(LABEL_DISTRIBUTION)
    assert distribution.num_rows > 0
    assert distribution.equals(tables["label_distribution"].select(LABEL_DISTRIBUTION.names))
    expected = Counter(row["label"] for row in rows(tables, "label_dataset"))
    assert {row["label"]: row["notes"] for row in distribution.to_pylist()} == dict(expected)


def test_metrics_run_on_the_sample(tables: dict) -> None:
    connection = duckdb.connect()
    for name in ("label_dataset", "notes_test_raw", "predictions"):
        connection.register(name, tables[name])

    def query(metric: str) -> list[tuple]:
        return connection.sql((REPO / "metrics" / f"{metric}.sql").read_text("utf-8")).fetchall()

    expected_coverage = tables["predictions"].num_rows / tables["notes_test_raw"].num_rows
    assert query("prediction_coverage")[0][0] == pytest.approx(expected_coverage)
    protocols = {row["label"] for row in rows(tables, "label_dataset")}
    assert query("protocol_count")[0][0] == len(protocols)
