task: label_protocols

Choose the radiology examination protocol that one clinical note asks for. Read the
note's findings, history and suspected condition, and pick the single protocol from
the allowed list below that a radiologist would schedule.

Return only one JSON object with these keys:

- `label`: exactly one protocol from the allowed list, spelled as it appears there.
- `confidence`: a number between 0 and 1.

Allowed protocols:
