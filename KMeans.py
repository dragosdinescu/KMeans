import json
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


kmeans = KMeans(n_clusters=4).fit(dataframe)
cluster_map = pd.DataFrame()
cluster_map['data_index'] = dataframe.index.values
cluster_map['cluster'] = kmeans.labels_
centroids = kmeans.cluster_centers_
print(centroids)

dataframe['cluster'] = kmeans.labels_

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(dataframe['x'], dataframe['y'], dataframe['z'], c= kmeans.labels_.astype(float),alpha=0.3)
ax.scatter3D(centroids[:, 0], centroids[:, 1],centroids[:, 2], c='red', s=50)

print(dataframe.cluster.unique())
fig = plt.figure()
ax = plt.axes(projection='3d')
for value in dataframe.cluster.unique():
    value_df = dataframe[dataframe.cluster == value]
    ax.scatter3D(value_df['x'], value_df['y'], value_df['z'], label=value)
plt.title(value)
plt.legend()
plt.show()

dataframe=dataframe[(dataframe['cluster'] != 1) & (dataframe['cluster'] != 3)]
print (dataframe)
kmeans = KMeans(n_clusters=10).fit(dataframe)
centroids = kmeans.cluster_centers_
print(centroids)

dataframe['cluster'] = kmeans.labels_

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(dataframe['x'], dataframe['y'], dataframe['z'], c= kmeans.labels_.astype(float),alpha=0.3)
ax.scatter3D(centroids[:, 0], centroids[:, 1],centroids[:, 2], c='red', s=50)
plt.show()
