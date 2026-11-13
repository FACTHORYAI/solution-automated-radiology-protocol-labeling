"""Prepare the labelled note set: deduplicate and drop rare protocol labels."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from automated_radiology_protocol_labeling import io
from automated_radiology_protocol_labeling.jobs._duck import align, run_query
from automated_radiology_protocol_labeling.schemas import LABEL_DATASET

INPUTS = ("notes_raw",)
OUTPUTS = ("label_dataset",)


@dataclass(frozen=True)
class Params:
    """Parameters for the note preparation."""

    min_label_count: int = 2


def run(inputs: dict[str, pa.Table], params: Params) -> dict[str, pa.Table]:
    """Return the distinct notes whose protocol label occurs often enough to train on."""
    if params.min_label_count < 1:
        message = f"Parameter min_label_count is {params.min_label_count}; it must be at least 1"
        raise ValueError(message)
    table = run_query(
        io.read_sql("prepare_notes"), inputs, {"min_label_count": params.min_label_count}
    )
    return {"label_dataset": align(table, LABEL_DATASET)}
