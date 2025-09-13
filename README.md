# distance_cluster
Purpose of these models is to replace static rules describing target data sub-populations with a dynamic model based on clustering of target data.

## baseline model: clustering.py
The baseline model consumes a diverse set of positive target data. It scales the numerical features within the input data to a mu = 0, sigma = 1 population, and applies Principal Component Analysis to decompose the data to its core coponents.
The model then applies KMeans clustering on the components data, to learn representations of the target data in the principal components' space.
In the next step, we apply any baseline data (random sample of the overall data) to learn the whole spectrum of data localization in the same space.
The fully fitted model can then consume any data and outputs the distance to the nearest learned target cluster in a 0-to-1 range, where 0 represents close proximity.
### process flow
Required: two datasets - one which only contains target-positive data (X_pos), and one which contains random training data (X). Any unseen data that we want to transform shall be denoted X_oot.
#### 1 - learning the target centroids
```
clustering = Clustering()
clustering._fit_cluster(X_pos)
```
#### 2 - learning the distance spectrum
```
clustering.fit(X)
```
#### 3 - transforming any new (unseen) data
```
distances = clustering.transform(X_oot)
```
The output is expected to be an array of distances between 0 and 1, with 0 denoting close proximity to the target data. 

## embeddigns model: deep_embeddings_clustering.py
This model represents an improvement over the baseline model in the following aspects:
a. clustering over artifacts acquired from supervised preprocessing
b. integration of categorical data through Weight of Evidence mapping
