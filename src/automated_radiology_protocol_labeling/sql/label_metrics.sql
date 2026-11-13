-- Precision, recall and F1 per protocol over the test notes.
WITH pairs AS (
    SELECT t.ground_truth AS actual,
           p.predicted_label AS predicted
    FROM notes_test_raw AS t
    JOIN predictions AS p
        ON p.note_id = t.note_id
),
labels AS (
    SELECT actual AS label FROM pairs
    UNION
    SELECT predicted AS label FROM pairs
),
counts AS (
    SELECT actual, predicted, count(*) AS notes
    FROM pairs
    GROUP BY actual, predicted
),
true_positive AS (
    SELECT actual AS label, notes AS true_positive FROM counts WHERE actual = predicted
),
support AS (
    SELECT actual AS label, sum(notes) AS support FROM counts GROUP BY actual
),
predicted_total AS (
    SELECT predicted AS label, sum(notes) AS predicted FROM counts GROUP BY predicted
),
metrics AS (
    SELECT l.label,
           coalesce(tp.true_positive, 0) AS true_positive,
           coalesce(s.support, 0) AS support,
           coalesce(pt.predicted, 0) AS predicted
    FROM labels AS l
    LEFT JOIN true_positive AS tp ON tp.label = l.label
    LEFT JOIN support AS s ON s.label = l.label
    LEFT JOIN predicted_total AS pt ON pt.label = l.label
)
SELECT label,
       true_positive,
       support,
       predicted,
       CASE WHEN predicted = 0 THEN 0.0
            ELSE CAST(true_positive AS DOUBLE) / predicted END AS precision,
       CASE WHEN support = 0 THEN 0.0
            ELSE CAST(true_positive AS DOUBLE) / support END AS recall,
       CASE
           WHEN true_positive = 0 OR predicted = 0 OR support = 0 THEN 0.0
           ELSE 2.0 * (CAST(true_positive AS DOUBLE) / predicted)
                * (CAST(true_positive AS DOUBLE) / support)
                / ((CAST(true_positive AS DOUBLE) / predicted)
                   + (CAST(true_positive AS DOUBLE) / support))
       END AS f1
FROM metrics
ORDER BY support DESC, label
