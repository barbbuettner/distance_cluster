# distance_cluster
## Background and purpose
In highly volatile environments (such as fraud prevention) a reliance on static, reactive rules frequently leads to minor behavioral shifts to escape a set of prevention rules, indicating a failure to close gaps permanently. While the occurrence of these behavioral shifts is expected, it is frequently unclear which aspect will be impacted, resulting in challenges of proactive implementation.
The idea of these models is to describe targeted patterns as clusters in a latent space, and apply prevention based on low distance to clusters. This removes the need to predict the exact nature of behavioral shifts, and moves from a purely reactive towards proactive evaluation.

## Model parameters
### all distance models
`n_clusters` : denotes the number of clusters in the clustering model(s). We generally recommend pre-assessment through silhouette scores, elbow plots, or other methods. Using a number that is too low may lead to high distance outputs even in target positive data. Using a number that is too high may lead to singular random data points in the input data skewing the output and mis-classifying target negative data as target positive.

`n_bins` : denotes the granularity of the distance percentiles. 10 bins is equivalent to single-digit distance output, 100 bins is equivalent to double-digit distance output, and so on. A higher number of bins requires slightly more memory capacity, while a lower number of bins may lead to poor output quality.
### baseline model only
`n_components` : we apply Principal Component Analysis during the preprocessing, the output of PCA is then used as input for the clustering model. Due to the curse of high dimensionality in clustering, we recommend n_components on a lower side, while ensuring not to lose too much insights. The input can either be an integer >= 1, fixing the number of components, or a float between 0 and 1, which creates the number of components based on tolerable information loss.
### embeddings model only
`target_rate_threshold` : for the embeddings model, we apply clustering on a random set of input data. To isolate the target dense clusters, this parameter defines the acceptable target density. Clusters with a target density (positive_events / total_events) below this threshold will be discarded.

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

## embeddings model: deep_embeddings_clustering.py
This model represents an improvement over the baseline model in the following aspects:
a. clustering over artifacts acquired from supervised preprocessing, and
b. integration of categorical data through Weight of Evidence mapping
Structurally, the idea is very similar: learn local representations of target positive and target distributed input data to obtain insights on proximity to any target data dense cluster.
### process flow
For this model we need training data X, and target data y. Any unseen data that we want to transform shall be denoted X_oot.
#### 1 - fitting the model
```
clustering = DeepEmbeddingsClustering()
clustering.fit(X, y)
```
#### 2 - transforming any new (unessen) data
```
distances = clustering.transform(X_oot)
```
The output is expected to be an array of distances between 0 and 1, with 0 denoting close proximity to a target-dense cluster.
