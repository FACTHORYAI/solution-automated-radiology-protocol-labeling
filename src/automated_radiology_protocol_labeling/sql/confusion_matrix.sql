-- Actual against predicted protocol for every test note.
SELECT t.ground_truth AS actual,
       p.predicted_label AS predicted,
       count(*) AS notes
FROM notes_test_raw AS t
JOIN predictions AS p
    ON p.note_id = t.note_id
GROUP BY t.ground_truth, p.predicted_label
ORDER BY actual, predicted
