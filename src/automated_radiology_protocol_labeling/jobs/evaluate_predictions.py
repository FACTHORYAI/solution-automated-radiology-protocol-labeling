"""Evaluate the predicted protocols against the ground truth of the test notes."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from automated_radiology_protocol_labeling import io
from automated_radiology_protocol_labeling.jobs._duck import align, run_query
from automated_radiology_protocol_labeling.schemas import (
    CONFUSION_MATRIX,
    EVALUATION_SUMMARY,
    LABEL_METRICS,
)

INPUTS = ("predictions", "notes_test_raw")
OUTPUTS = ("evaluation_summary", "label_metrics", "confusion_matrix")


@dataclass(frozen=True)
class Params:
    """Parameters for the evaluation."""


def run(inputs: dict[str, pa.Table], params: Params) -> dict[str, pa.Table]:
    """Return the accuracy, per-protocol metrics and the confusion matrix."""
    summary = run_query(io.read_sql("evaluation_summary"), inputs)
    metrics = run_query(io.read_sql("label_metrics"), inputs)
    matrix = run_query(io.read_sql("confusion_matrix"), inputs)
    return {
        "evaluation_summary": align(summary, EVALUATION_SUMMARY),
        "label_metrics": align(metrics, LABEL_METRICS),
        "confusion_matrix": align(matrix, CONFUSION_MATRIX),
    }
