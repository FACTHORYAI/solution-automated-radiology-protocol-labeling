"""Parse and validate model answers before they become table rows."""

from __future__ import annotations

import json


def parse_object(answer: str) -> dict:
    """Parse a model answer that must be one JSON object."""
    try:
        item = json.loads(answer)
    except json.JSONDecodeError as error:
        message = f"model answer is not JSON: {error.msg} at position {error.pos}"
        raise ValueError(message) from error
    if not isinstance(item, dict):
        message = "model answer is not a JSON object"
        raise ValueError(message)
    return item


def validate_label_item(item: dict, vocabulary: set[str]) -> str:
    """Return the reason the prediction is rejected, or "" when it is valid."""
    label = item.get("label")
    if not isinstance(label, str) or not label.strip():
        return "missing or empty label"
    if label not in vocabulary:
        return f"label {label!r} is not one of the known protocols"
    confidence = item.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return "confidence is not a number"
    if not 0 <= float(confidence) <= 1:
        return "confidence is not between 0 and 1"
    return ""
