"""The jobs of this solution: plain functions from named tables to named tables."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from automated_radiology_protocol_labeling.jobs import (
    evaluate_predictions,
    label_protocols,
    label_summary,
    prepare_notes,
)


@dataclass(frozen=True)
class JobSpec:
    """One job: its callable, its parameter class and its input and output tables."""

    name: str
    description: str
    run: Callable[..., dict[str, pa.Table]]
    params_cls: type
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    needs_model: bool = False


REGISTRY: dict[str, JobSpec] = {
    spec.name: spec
    for spec in (
        JobSpec(
            name="prepare_notes",
            description="Prepare the labelled note set: deduplicate and drop rare protocol labels.",
            run=prepare_notes.run,
            params_cls=prepare_notes.Params,
            inputs=prepare_notes.INPUTS,
            outputs=prepare_notes.OUTPUTS,
        ),
        JobSpec(
            name="label_protocols",
            description="Label test notes with the radiology protocol they call for.",
            run=label_protocols.run,
            params_cls=label_protocols.Params,
            inputs=label_protocols.INPUTS,
            outputs=label_protocols.OUTPUTS,
            needs_model=True,
        ),
        JobSpec(
            name="evaluate_predictions",
            description="Accuracy, per-protocol metrics and confusion matrix over the test notes.",
            run=evaluate_predictions.run,
            params_cls=evaluate_predictions.Params,
            inputs=evaluate_predictions.INPUTS,
            outputs=evaluate_predictions.OUTPUTS,
        ),
        JobSpec(
            name="label_summary",
            description="How the labelled notes distribute over the protocol labels.",
            run=label_summary.run,
            params_cls=label_summary.Params,
            inputs=label_summary.INPUTS,
            outputs=label_summary.OUTPUTS,
        ),
    )
}


def param_values(spec: JobSpec, args: Any) -> Any:
    """Build the job's Params instance from parsed command-line arguments."""
    values = {
        field.name: getattr(args, field.name)
        for field in spec.params_cls.__dataclass_fields__.values()
    }
    return spec.params_cls(**values)


def call_run(
    spec: JobSpec, inputs: dict[str, pa.Table], params: Any, model: Any
) -> dict[str, pa.Table]:
    """Call the job's run with the arguments its signature asks for."""
    if spec.needs_model:
        return spec.run(inputs=inputs, params=params, model=model)
    return spec.run(inputs=inputs, params=params)
