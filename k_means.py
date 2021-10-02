import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.manifold import MDS

filename = "lollipops.dat"

df = pd.read_table(filename, sep=";", header=0)


data = df.iloc[:, [1, 2, 3]].values

wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 15), wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of cluster (k)')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=3)


k = kmeans.fit_predict(data)

df['label'] = k

print(df)

cmd = MDS(n_components=2)
trans = cmd.fit_transform(data)


print(trans.shape)

plt.scatter(trans[k == 0, 0], trans[k == 0, 1], s=10, c='red', label='Cluster 1')
plt.scatter(trans[k == 1, 0], trans[k == 1, 1], s=10, c='blue', label='Cluster 2')
plt.scatter(trans[k == 2, 0], trans[k == 2, 1], s=10, c='green', label='Cluster 3')
plt.show()

writer = pd.ExcelWriter('123.xlsx')

from statistics import mode
for i, group in df.groupby('label'):
    print('=' * 10)
    print('cluster {}'.format(i))
    print(group.iloc[:, :-5].values)

    val = [np.sum(x) for x in group.iloc[:, :-5].values]
    print(np.mean(val))
    print(np.mean(group['V1']))
    print(np.mean(group['V2']))
    print(mode(group['V3']))
    print(mode(group['V4']))
