"""
Created on Fri Feb 10 09:07:28 2023

@author: lionel
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import time

duration =[]
sizes = range(500, 15000, 1000)
for n in sizes:
    

    X, y = make_blobs(n_samples=n, n_features=500, centers=5, cluster_std=1.0, shuffle=False)
    
    
    start = time.time()
    model = AgglomerativeClustering(n_clusters=5, affinity='precomputed', memory=None, connectivity=None, compute_full_tree='auto', linkage='single', distance_threshold=None, compute_distances=False)
    
    model.fit_predict(np.sqrt(2 - np.corrcoef(X)**2))
    
    stop = time.time()
    
    duration.append(stop - start)
    print(n)
    
plt.figure()
plt.plot(sizes,duration)
plt.scatter(sizes,duration, s=5)
plt.xlabel('dataset size')
plt.ylabel('duration')
