-- How many of the test notes the model labelled with the right protocol.
WITH pairs AS (
    SELECT t.ground_truth AS actual,
           p.predicted_label AS predicted
    FROM notes_test_raw AS t
    JOIN predictions AS p
        ON p.note_id = t.note_id
)
SELECT CAST(SUM(CASE WHEN actual = predicted THEN 1 ELSE 0 END) AS DOUBLE) / count(*) AS accuracy,
       SUM(CASE WHEN actual = predicted THEN 1 ELSE 0 END) AS correct,
       count(*) AS notes
FROM pairs
