import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class FakeKMeans:
    def __init__(self, center):
        self.cluster_centers_ = np.array([center])
        self.n_clusters = 1
    
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)
    
    def transform(self, X):
        return np.linalg.norm(X - self.cluster_centers_[0], axis=1).reshape(-1, 1)


class DeepEmbeddingsClustering:
  def __init__(self, n_clusters=8, n_bins=1000, random_state=23, target_rate_threshold=0.8):
    self.n_clusters = n_clusters
    self.n_bins = n_bins
    self.target_rate_threshold = target_rate_threshold
    self.random_state = random_state

  def get_num_cat(self, X: pd.DataFrame, copy=False):
    if copy:
      X = X.copy()

    num_cols = []
    cat_cols = []
    for col in X.columns:
      try:
        X[col] = X[col].astype(float).fillna(-1)
        num_cols.append(col)
      except Exception:
        X[col] = X[col].fillna("").astype(str)
        cat_cols.append(col)

    self.numerical_columns = num_cols
    self.categorical_columns = cat_cols
    return X

  def get_cat_woe(self, X: pd.DataFrame, y: pd.Series):
    woe_map = {}
    for col in getattr(self, "categorical_columns", []):
      woe_map_col = {}
      for colval in set(X[col]):
        pos_events = sum(((y == 1) & (X[col] == colval)).astype(int))
        neg_events = sum(((y == 0) & (X[col] == colval)).astype(int))
        events = sum((X[col] == colval).astype(int))
        perc_pos = pos_events / events
        perc_neg = neg_events / events
        woe_map_col[colval] = np.log(perc_pos / perc_neg)
      woe_map[col] = woe_map_col
    self.woe_map = woe_map

  def preprocess_num(self, X: pd.DataFrame):
    if hasattr(self, "preprocessor_num"):
      return self.preprocessor_num.transform(X[self.numerical_columns])
    if not hasattr(self, "numerical_columns"):
      print("Need to assign numerical and categorical values first")
      return None

    imputer = SimpleImputer(strategy="constant", fill_value=-1)
    scaler = StandardScaler()
    self.preprocessor_num = Pipeline([("imputer", imputer), ("scaler", scaler)]).fit(X[self.numerical_columns])
    return self.preprocessor_num.transform(X[self.numerical_columns])

  def preprocess_cat(self, X: pd.DataFrame):
    if not hasattr(self, "woe_map"):
      print("Need to collect woe map first")
      return None
    X_cat_transformed = X[self.categorical_columns].copy()
    X_cat_transformed = X_cat_transformed.replace({col: self.woe_map[col] for col in self.categorical_columns})
    X_cat_transformed = X_cat_transformed.fillna(0.0)
    return np.array(X_cat_transformed)

  def forward_to_penultimate(self, X: np.ndarray):
    if not hasattr(self, "dec_model") or not hasattr(self.dec_model, "coefs_"):
      print("Need to fit dec model first")
      return None

    reconstructed_data = X.copy()
    for i in range(len(self.dec_model.coefs_) - 1):
      reconstructed_data = (np.matmul(reconstructed_data, self.dec_model.coefs_[i]) + self.dec_model.intercepts_[i])
    return reconstructed_data

  def fit(self, X: pd.DataFrame, y: pd.Series):
    if hasattr(self, 'percentiles'):
      return self

    X = self.get_num_cat(X)
    self.get_cat_woe(X, y)
    X_num = self.preprocess_num(X)
    X_cat = self.preprocess_cat(X)
    X_transformed = np.hstack((X_num, X_cat))

    self.dec_model = MLPClassifier(hidden_layer_sizes=(256, 64, 8), random_state=self.random_state)
    self.dec_model.fit(X_transformed, y)
    X_reconstructed = self.forward_to_penultimate(X_transformed)
    self.clustering_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
    self.clustering_model.fit(X_reconstructed)

    labels = self.clustering_model.predict(X_reconstructed)
    exclusion_clusters = []
    for lbl in list(set(labels)):
      pos_events = sum(((labels == lbl) & (y == 1)).astype(int))
      neg_events = sum(((labels == lbl) & (y == 0)).astype(int))
      if pos_events / (pos_events + neg_events) < self.target_rate_threshold:
        exclusion_clusters.append(lbl)

    if len(exclusion_clusters) == len(list(set(labels))):
      print("No cluster meets the fraud rate criteria. Stopping modification")
      return self

    final_labels = np.where(np.isin(labels, exclusion_clusters), -1, labels)
    remaining_centers = self.clustering_model.cluster_centers_[~np.isin(np.arange(self.clustering_model.n_clusters), exclusion_clusters)]

    if len(exclusion_clusters) + 1 == len(list(set(labels))):
      self.clustering_model = FakeKMeans(remaining_centers)
      return self
    
    self.n_clusters = remaining_centers.shape[0]
    self.clustering_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=1, init=remaining_centers).fit(X_reconstructed)

    dists = self.clustering_model.transform(X_reconstructed)
    nearest_dist = dists.min(axis=1)
    self.percentiles = np.searchsorted(np.percentile(nearest_dist, np.linspace(0, 100, self.n_bins)))
    return self

  def transform(self, X:pd.DataFrame):
    if not hasattr(self, 'percentiles'):
      print("Must fit the model first")
      return self
    X_num = self.preprocess_num(X)
    X_cat = self.preprocess_cat(X)
    X_transformed = np.hstack((X_num, X_cat))
    X_reconstructed = self.forward_to_penultimate(X_transformed)
    dists = self.clustering_model.transform(X_reconstructed)
    nearest_dist = dists.min(axis=1)
    dist_percentile_idx = np.searchsorted(self.percentiles, nearest_dist)
    dist_percentile = dist_percentile_idx / self.n_bins
    return dist_percentile
