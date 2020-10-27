
import json
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

def getdist( p1, p2 ):
   return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)+((p1[2]-p2[2])**2))

with open('participant_101.json') as f:
    data = json.load(f)
index = 0
dataframe = pd.DataFrame(columns=["x", "y", "z"])
distances = []
previous = [-1, -1, -1]

with open('output.txt', 'a') as f:
    for line in data:
        gp = line['Gaze Position 3D']['gp3']
        if previous == [-1, -1, -1]:
            previous = [gp[0], gp[1], gp[2]]
        dataframe.loc[index] = [gp[0], gp[1], gp[2]]
        newdist = getdist(previous, [gp[0], gp[1], gp[2]])
        distances.insert(index, newdist)
        previous = [gp[0], gp[1], gp[2]]
        index = index + 1
        f.write(str(previous[0]))
        f.write(" ")
        f.write(str(previous[1]))
        f.write(" ")
        f.write(str(previous[2]))
        f.write(" ")
        f.write(str(newdist))
        f.write("\n")


print(dataframe)
plt.plot(distances)
plt.show()

kmeans = KMeans(n_clusters=4).fit(dataframe)
cluster_map = pd.DataFrame()
cluster_map['data_index'] = dataframe.index.values
cluster_map['cluster'] = kmeans.labels_
centroids = kmeans.cluster_centers_

#print(centroids)
#print(cluster_map[cluster_map.cluster == 3])
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.scatter3D(dataframe['x'], dataframe['y'], dataframe['z'], c=kmeans.labels_.astype(float), alpha=0.3)
#ax.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=50)

#plt.show()
