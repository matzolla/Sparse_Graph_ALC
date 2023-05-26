from Graph_ALC import graph_alc


import numpy as np
import random
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.sparse.csgraph import connected_components
import time
######### test examples ########################



n_samples = 100
n_features = 1000
n_clusters= 12
n_neighbors = 10
X, y = make_blobs(n_samples = n_samples, n_features = n_features,centers=n_clusters, cluster_std=1, shuffle=False)



graph = kneighbors_graph(X, n_neighbors, metric='cosine', mode='distance')
graph_mat = graph.todense()
sol1  = graph_alc(np.array(graph_mat))

sol1.initializer()

## computing the time
start=time.time()
graphnet=sol1.Alc()
stop= time.time()

G = nx.from_numpy_matrix(graphnet)  
plt.figure()
nx.draw(G, with_labels=False,node_size=20,node_color=y) 
plt.show()


n_components, labels = connected_components(csgraph=nx.to_scipy_sparse_matrix(G), directed=False, return_labels=True)

print("ground truth: {}".format(y))
print("predicted labels: {}".format(labels))

print("The adjsuted rand score is {}".format(adjusted_rand_score(y,labels)))

print("The runtime is:  {}".format((stop-start)))



graph = kneighbors_graph(X, n_neighbors, metric='cosine', mode='distance')
G = nx.from_scipy_sparse_matrix(graph)

plt.figure()
nx.draw(G, node_size =20, node_color = y)
plt.show()