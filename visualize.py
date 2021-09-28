import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def plot(y_pred, X_train, title='Title'):
    y_unique = np.unique(y_pred)
    num_clusters=len(y_unique)
    n = X_train.shape[1]

    plt.figure(figsize=(17, 8))

    for i, yi in enumerate(y_unique):
        plt.subplot(2, (num_clusters // 2) + 1, 1 + i)
        
        for xx in X_train[y_pred == yi]:
            plt.plot(xx, "k-", alpha=.2)
        
        # plot mean trade session for cluster
        plt.plot(X_train[y_pred == yi].mean(axis=0), "r-")
        plt.xlim(0, n)
        plt.ylim(-4, 4)
        plt.ylabel('Normalized price')

        
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                transform=plt.gca().transAxes)
        if i == 1:
            plt.title(title)