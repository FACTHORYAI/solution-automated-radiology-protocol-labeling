"""How the labelled notes distribute over the protocol labels."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from automated_radiology_protocol_labeling import io
from automated_radiology_protocol_labeling.jobs._duck import align, run_query
from automated_radiology_protocol_labeling.schemas import LABEL_DISTRIBUTION

INPUTS = ("label_dataset",)
OUTPUTS = ("label_distribution",)


@dataclass(frozen=True)
class Params:
    """Parameters for the label summary."""


def run(inputs: dict[str, pa.Table], params: Params) -> dict[str, pa.Table]:
    """Return the number of training notes per protocol label."""
    table = run_query(io.read_sql("label_distribution"), inputs)
    return {"label_distribution": align(table, LABEL_DISTRIBUTION)}
