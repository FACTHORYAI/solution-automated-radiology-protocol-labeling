-- How the training notes distribute over the protocol labels.
SELECT label,
       count(*) AS notes
FROM label_dataset
GROUP BY label
ORDER BY notes DESC, label
