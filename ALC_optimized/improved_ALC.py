'''Project: “Agglomerative Likelihood Clustering”
Author : Lionel Yelibi, 2019, University of Cape Town.
Copyright SPC, 2019, 2020, 2021
Potts Model Clustering.
Agglomerative Likelihood Clustering
See pre-print: https://arxiv.org/abs/1908.00951
GNU GPL
This file is part of ALC
ALC is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
ALC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

it might help : https://github.com/shubhamjha97/hierarchical-clustering/blob/master/Agglomerative/agglomerative.py

'''


import numpy as np
from sklearn.datasets import make_blobs
from numba import jit

''' the cluster function:
    compute the likelihood which occurs when two objects are clustered or
    the object own likelihood. By design if a cluster only containts one object
    its likelihood will be 0'''
@jit(nopython=True)
def clus_lc(gij,gii,gjj, ns=2):
    ''' variables description:
        ns is the size of the cluster.
        cs is the intracluster correlation.
        gij, gii, and gjj respectively are
        correlations relating to interactions between objects i and j, and
        their respective self-correlations. self-correlations (i.e. gii, gjj)
        are 1 for individual objects and >1 for clusters'''
        
    if ns==1:
        return 0
    ''' intracluster correlation'''   
    cs = 2*gij+gii+gjj-1e-3
    ''' relatively low cs means noisy and suboptimal clusters.
    The coupling parameter gs (see paper) isn't not defined'''
    if cs<=ns:
        return 0
    return 0.5*(np.log(ns/cs) +  (ns - 1)*np.log( (ns**2 - ns) / ( ns**2 - cs) ))

''' aspc only requires a correlation matrix as input:
    here we convert the correlation to a dictionary for convenience. adding new
    entries in a dict() is much faster than editing a numpy matrix'''
def alc(G, cn= None):
    
    N = len(G)
    gdic = dict(enumerate(dict(enumerate(row)) for row in G))
    # values=[0]*N 
    # clike = dict(enumerate(values))#{ i : 0 for i in range(N)}
    del G
    
    ''' tracker is dictionary which stores the objects member of the same clusters.
        the data is stored as strings: i.e. cluster 1234 contains objects 210 & 890
        which results in tracker['1234'] == '210_890' '''
    tracker = { i:[i] for i in range(N) }
    
    # ''' the cluster size ns is stored in the ns_ array'''
    # ns_ = [1]*N
    
    ''' Create a list of object indices 'other_keys': at every iteration one object
     is clustered and removed from the list. It is also removed if no suitable
     optimal destination is found.'''
    other_keys = list(range(N))
    
    ''' the operation stocks once there is only one object left to cluster as we
    need two objects at the very least.'''
    while len(tracker) != cn:
        
        ''' a random initialization:
            pick a object 'node' at random to start clustering,
            this might have a consequence on the final result depending on the data.
            then loop through the other objects using 'nbor' and costs to store
            the likelihood resulting from clustering 'node' to the objects in
            'nbor'.
            indices: stores the indices which are combinations of node and others.
            costs: stores the cost which compute the difference between the likelihood
            of the resulting cluster and the sum of the two individual objects forming
            the result cluster.
            '''
        ''' the routine uses other_keys and removes elements everytime they are clustered
        or can't be clustered anymore. If a cluster number is not provided the routine
        stops there. If one is then it continues by looking at the elements in the
        optimal cluster solution (tracker) and continues merging until the preset
        number of clusters is met'''


        if len(other_keys)>1:
            node = np.random.choice(other_keys)
        else:
            if len(tracker) != 1:
                node = np.random.choice(list(tracker.keys()))
            else:
                cn = 1
                continue
        nbor = list(tracker.keys())
        nbor.remove(node)
        costs = [0]*len(nbor) 
        indices = [(node,key) for key in nbor]
        node_lc = clus_lc(0,gdic[node][node],0,ns = len(tracker[node]))
  
        for k,(i,j) in enumerate(indices):
            costs[k] = clus_lc(gdic[i][j],gdic[i][i],gdic[j][j],ns=len(tracker[i])+len(tracker[j])) - (node_lc+clus_lc(0,gdic[j][j],0,ns = len(tracker[j])))
            
        ''' find the optimal cost which will be the object clustered with node'''
        next_merge = np.argmax(costs)
        
        
        ''' stopping conditions '''
        if costs[next_merge]<=0:
            if len(other_keys)>1:
                ''' if no cost is positive then this node cannot be clustered further
                and must be removed from the list'''
                other_keys.remove(node)
                continue
            elif not cn:
                ''' if no cluster number is provided then the routine has completed
                and tracker is the final solution'''
                cn = len(tracker)
                continue
            elif cn:
                ''' if a cluster number is provided the routine continues and keeps
                merging'''
                pass
        ''' on the other hand, the largest positive cost is the designated
        object 'label_b' clustered to node which here is stored as 'label_a'.
        new clusters 'new_label' take values superior to N.
        tracker, as previously explained, stores joined strings of the clusters
        contents'''
        
        label_a = node
        label_b = indices[next_merge][1]
        new_label = list(tracker.keys())[-1]+1
        # clike[new_label] = clus_lc(gdic[label_a][label_b],gdic[label_a][label_a],gdic[label_b][label_b],ns=len(tracker[label_a])+len(tracker[label_b])) 
        # del clike[label_a]
        # del clike[label_b]
        ''' removes merged elements and update others with the new cluster.
        only do it when a positive cost is found.'''
        if costs[next_merge]>0:
            other_keys = list(tracker.keys())
            other_keys.remove(label_a)
            other_keys.remove(label_b)
            other_keys.append(new_label)
    
        
        ''' Once a cluster is formed, the correlation matrix gdic and tracker need to
        be updated with the new cluster and the cluster size must be updated with ns_'''
        nbor.remove(label_b)
        tracker[new_label]=tracker[label_a] + tracker[label_b]
        gdic[new_label]=  dict()
        gdic[new_label][new_label] = 2*gdic[label_a][label_b] + gdic[label_a][label_a] + gdic[label_b][label_b]
    
        for key in nbor:
            gdic[new_label][key] = gdic[label_a][key] + gdic[label_b][key]
            gdic[key][new_label] =gdic[new_label][key] 
    
        del tracker[label_a]
        del tracker[label_b]
    
    
    ''' create the final clustering array:
        tracker contains the cluster memberships but as a dictionary
        we create a numpy array where stocked are labeled with the same number
        if they belong to the same cluster, and 0 if unclustered'''
        
    solution = np.zeros(N,dtype=int)

    for k,cluster in enumerate(tracker.keys()):
        cluster_members = tracker[cluster]
        solution[cluster_members] = k+1
    return solution

sol1  = alc(np.corrcoef(data))
