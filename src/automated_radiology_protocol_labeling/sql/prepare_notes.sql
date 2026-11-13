-- One row per distinct note, keeping only labels that occur at least $min_label_count
-- times (the training-data filter of the original pipeline used 50).
WITH counts AS (
    SELECT label, count(*) AS label_notes
    FROM notes_raw
    WHERE label IS NOT NULL AND label <> ''
      AND text IS NOT NULL AND text <> ''
    GROUP BY label
),
distinct_notes AS (
    SELECT min(note_id) AS note_id,
           text,
           min(label) AS label
    FROM notes_raw
    WHERE label IS NOT NULL AND label <> ''
      AND text IS NOT NULL AND text <> ''
    GROUP BY text
)
SELECT note_id, text, label
FROM distinct_notes
WHERE label IN (SELECT label FROM counts WHERE label_notes >= $min_label_count)
ORDER BY label, note_id
