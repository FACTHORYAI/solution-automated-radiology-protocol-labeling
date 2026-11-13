# Automated Radiology Protocol Labeling

Label radiology examination protocols from clinical notes, and measure how well the labelling performs.

## Start here

1. Look at the sample tables in `data/sample` (synthetic): `notes_raw` and `notes_test_raw` hold labelled and held-out notes across ten protocols.
2. Build the training set with `prepare_notes` (deduplicates notes and drops protocols too rare to learn), then label the test notes with `label_protocols`.
3. Judge the result:

| Question | Job |
| --- | --- |
| Which protocol does this note call for? | `label_protocols` |
| How accurate is the labelling overall? | `evaluate_predictions` |
| Which protocols are confused with which? | `evaluate_predictions` (`confusion_matrix`, `label_metrics`) |
| How balanced is the training set? | `label_summary` |

Named measures for prediction coverage and protocol count live in `metrics/`.

## Notes

- The sample data is synthetic; real patient notes never enter this project.
- `label_protocols` needs a text model behind `automated_radiology_protocol_labeling.text.interface`; a recorded stand-in exists for tests and demos. The other jobs run without any model.
- A predicted protocol is a scheduling suggestion for a radiologist to review, not an order.
- Results are for research and analytics, not clinical decisions.
