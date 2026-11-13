-- Share of the test notes that received a protocol prediction.
SELECT CAST((SELECT count(*) FROM predictions) AS DOUBLE)
       / (SELECT count(*) FROM notes_test_raw) AS prediction_coverage
