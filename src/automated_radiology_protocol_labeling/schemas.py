"""PyArrow schemas for every table the jobs read or write."""

import pyarrow as pa

NOTES_RAW = pa.schema(
    [
        pa.field("note_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("label", pa.string()),
    ]
)

NOTES_TEST_RAW = pa.schema(
    [
        pa.field("note_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("ground_truth", pa.string()),
    ]
)

LABEL_DATASET = pa.schema(
    [
        pa.field("note_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("label", pa.string()),
    ]
)

PREDICTIONS = pa.schema(
    [
        pa.field("note_id", pa.string()),
        pa.field("predicted_label", pa.string()),
        pa.field("confidence", pa.float64()),
    ]
)

REJECTED = pa.schema(
    [
        pa.field("note_id", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("model_answer", pa.string()),
    ]
)

EVALUATION_SUMMARY = pa.schema(
    [
        pa.field("accuracy", pa.float64()),
        pa.field("correct", pa.int64()),
        pa.field("notes", pa.int64()),
    ]
)

LABEL_METRICS = pa.schema(
    [
        pa.field("label", pa.string()),
        pa.field("true_positive", pa.int64()),
        pa.field("support", pa.int64()),
        pa.field("predicted", pa.int64()),
        pa.field("precision", pa.float64()),
        pa.field("recall", pa.float64()),
        pa.field("f1", pa.float64()),
    ]
)

CONFUSION_MATRIX = pa.schema(
    [
        pa.field("actual", pa.string()),
        pa.field("predicted", pa.string()),
        pa.field("notes", pa.int64()),
    ]
)

LABEL_DISTRIBUTION = pa.schema(
    [
        pa.field("label", pa.string()),
        pa.field("notes", pa.int64()),
    ]
)
