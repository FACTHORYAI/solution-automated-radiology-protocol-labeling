"""Label test notes with the radiology protocol they call for."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from automated_radiology_protocol_labeling import io
from automated_radiology_protocol_labeling.jobs._duck import align
from automated_radiology_protocol_labeling.schemas import PREDICTIONS, REJECTED
from automated_radiology_protocol_labeling.text.interface import TextModel
from automated_radiology_protocol_labeling.text.parse import parse_object, validate_label_item

INPUTS = ("notes_test_raw", "label_dataset")
OUTPUTS = ("predictions", "rejected_labels")


@dataclass(frozen=True)
class Params:
    """Parameters for protocol labelling."""


def run(inputs: dict[str, pa.Table], params: Params, model: TextModel) -> dict[str, pa.Table]:
    """Return the predicted protocol per test note and every answer that failed validation."""
    notes = inputs["notes_test_raw"]
    vocabulary = sorted({row["label"] for row in inputs["label_dataset"].to_pylist()})
    if not vocabulary:
        message = "label_dataset has no labels; run prepare_notes before label_protocols"
        raise ValueError(message)
    instructions = io.read_prompt("label_protocols") + "\n".join(
        f"- {label}" for label in vocabulary
    )
    allowed = set(vocabulary)
    rows: list[dict] = []
    rejected: list[dict] = []
    for note in notes.to_pylist():
        answer = model.complete(instructions, note["text"])
        try:
            item = parse_object(answer)
            reason = validate_label_item(item, allowed)
        except ValueError as error:
            reason = str(error)
        if reason:
            rejected.append({"note_id": note["note_id"], "reason": reason, "model_answer": answer})
            continue
        rows.append(
            {
                "note_id": note["note_id"],
                "predicted_label": item["label"],
                "confidence": float(item["confidence"]),
            }
        )
    return {
        "predictions": align(pa.Table.from_pylist(rows, schema=PREDICTIONS), PREDICTIONS),
        "rejected_labels": align(pa.Table.from_pylist(rejected, schema=REJECTED), REJECTED),
    }
