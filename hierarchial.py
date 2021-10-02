import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

from sklearn.cluster import AgglomerativeClustering

filename = "lollipops.dat"

link_method = 'ward'

df = pd.read_table(filename, sep=";", header=0)

data = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].values

linked = linkage(data, method=link_method)

plt.figure(figsize=(10, 6))
plt.title('Hierarchial Clustering Dendrogram')
plt.xlabel('Cities')
plt.ylabel('Distance')
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True,
           truncate_mode='lastp')
plt.axhline(y=45, c='k')
plt.show()

# elbow

hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage=link_method)


def wss_calculation(K, data):
    WSS = []
    for i in range(K):
        cluster = AgglomerativeClustering(n_clusters=i + 1, affinity='euclidean', linkage=link_method)
        cluster.fit_predict(data)
        # cluster index
        label = cluster.labels_
        wss = []
        for j in range(i + 1):
            # extract each cluster according to its index
            idx = [t for t, e in enumerate(label) if e == j]
            cluster = data[idx,]
            # calculate the WSS:
            cluster_mean = cluster.mean(axis=0)
            distance = np.sum(np.abs(cluster - cluster_mean) ** 2, axis=-1)
            wss.append(sum(distance))
        WSS.append(sum(wss))
    return WSS


WSS = wss_calculation(15, data)

cluster_range = range(1, 16)

plt.figure(figsize=(10, 5))
plt.title('Optimal number of cluster')
plt.xlabel('Number of cluster (k)')
plt.ylabel('Total intra-cluster variation')
plt.plot(cluster_range, WSS, marker="x")
plt.show()

fig, ax = plt.subplots(figsize=(20, 10))

ax = dendrogram(linked)

plt.show()

label = fcluster(linked, 45, criterion='distance')
print(np.unique(label))

df.loc[:, 'label'] = label

for i, group in df.groupby('label'):
    print('=' * 10)
    print('cluster {}'.format(i))
    print(group.iloc[:, :-5].values)

    val = [np.sum(x) for x in group.iloc[:, :-5].values]
    print(np.mean(val))
    print(np.mean(group['V1']))
    print(np.mean(group['V2']))
