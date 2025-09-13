# distance_cluster
Purpose of these models is to replace static rules describing target data sub-populations with a dynamic model based on clustering of target data.

## baseline model: clustering.py
The baseline model consumes a diverse set of positive target data. It scales the numerical features within the input data to a mu = 0, sigma = 1 population, and applies Principal Component Analysis to decompose the data to its core coponents.
The model then applies KMeans clustering on the components data, to learn representations of the target data in the principal components' space.
In the next step, we apply any baseline data (random sample of the overall data) to learn the whole spectrum of data localization in the same space.
The fully fitted model can then consume any data and outputs the distance to the nearest learned target cluster in a 0-to-1 range, where 0 represents close proximity.
