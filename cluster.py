import numpy as np

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN


def normalize(X_train):
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
    return X_train


def clustering(X_train, metric='dtw', num_clusters=5):
    seed = 0
    np.random.seed(seed)

    X_train = normalize(X_train)
    
    if metric == 'dtw':
        X = X_train
        clust = TimeSeriesKMeans(
            n_clusters=num_clusters,
            n_init=2,
            metric="dtw",
            max_iter_barycenter=10,
            random_state=seed
        )

    elif metric == 'MAE':
        X = X_train.squeeze()
        # MAE is similar to L1-norm
        clust = DBSCAN(metric='l1', eps=30, min_samples=5)

    elif metric == 'cosine':
        X = X_train.squeeze()
        X = preprocessing.normalize(X)
        # Euclidean distance with normalized vectors (length=1) is similar to cosine distance
        clust = KMeans(n_clusters=num_clusters)

    elif metric == 'l2':
        # Euclidean distance
        X = X_train
        clust = TimeSeriesKMeans(
            n_clusters=num_clusters, 
            random_state=seed
        )
    
    else: 
        raise(NotImplementedError)

    y_pred = clust.fit_predict(X)
    return y_pred