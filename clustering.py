import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class Clustering:
  def __init__(self, n_clusters=8, n_components=8, n_bins=1000, random_state=23):
    self.n_clusters = n_clusters
    self.n_components = n_components
    self.n_bins = n_bins
    self.random_state = random_state

  def _fit_cluster(self, X:pd.DataFrame):
    """fit KMeans model with target data to construct cluster centroids"""
    X_decomposed = self._preprocess(X)
    self.clustering_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(X_decomposed)
    return self

  def fit(self, X:pd.DataFrame):
    """fit the model with baseline data (healthy mix of positive and negative)"""
    if not hasattr(self, 'clustering_model'):
      print("Must fit the KMeans model first!")
      return self
    X_decomposed = self._preprocess(X)
    dists = self.clustering_model.transform(X_decomposed)
    nearest_dist = dists.min(axis=1)
    self.percentiles = np.percentile(nearest_dist, np.linspace(0, 100, self.n_bins))
    return self

  def transform(self, X:pd.DataFrame):
    if not hasattr(self, 'percentiles'):
      print("Must fit the model first")
      return self
    X_decomposed = self._preprocess(X)
    dists = self.clustering_model.transform(X_decomposed)
    nearest_dist = dists.min(axis=1)
    dist_percentile_idx = np.searchsorted(self.percentiles, nearest_dist)
    dist_percentile = dist_percentile_idx / self.n_bins
    return dist_percentile
  
  def _preprocess(self, X:pd.DataFrame):
    if hasattr(self, 'preprocessor'):
      return self.preprocessor.transform(X)
    numerical_features = []
    for col in X.columns:
      try:
        X[col] = X[col].apply(float)
        numerical_features.append(col)
      except:
        pass
    imputer = SimpleImputer(strategy='constant', fill_value=-1)
    scaler = StandardScaler()
    pca = PCA(n_components=self.n_components)
    self.preprocessor = Pipeline([('imputer', imputer), ('scaler', scaler), ('pca', pca)]).fit(X[numerical_features])
    return self.preprocessor.transform(X[numerical_features])
  
