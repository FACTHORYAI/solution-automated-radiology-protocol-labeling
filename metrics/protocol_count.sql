-- How many distinct radiology protocols the training set covers.
SELECT count(DISTINCT label) AS protocols
FROM label_dataset
