# All rights reserved.
'''
Project: “Graph Agglomerative Likelihood Clustering”
Author : Allassan Tchangmena, Lionel Yelibi, 2023, Spin City Lab.
'''

import numpy as np
import random
from collections import defaultdict

class graph_alc:

    def __init__(self,corr_mat):
        
        # init

        '''
        Variable description (global variables)
        -------------
        corr_mat          : a similarity metric obtain from a KNN graph
        keys              : a dictionary which keep tracks of nodes (as key) and likelihood (as value)
        likelihood        : dictionary which keep tracks of nodes in a community (as key) and the likelihood (as value)
        candidates        : a group of community (keep track of a list of list, each list been a community)
        weights           : a dictionary of nodes as key and value as weight (similarity between to linked nodes)
        seen              : lookup set to check if the node has already been added to a community
        curr_node         : keep track of the node that is being merged in a communtiy (current node)
        delta_lc          : compute the change in likelihood
        result            : a NxN matrix that is updated continuously

        '''

        self.corr_mat=corr_mat
        self.keys=defaultdict()
        self.likelihood={}
        self.edges={}
        self.candidates=[]
        self.weights={}
        self.seen=set()
        self.curr_node=None
        self.delta_lc= -float('inf')
        self.result=np.zeros((self.corr_mat.shape[0],self.corr_mat.shape[0]))

    def initializer(self):

        '''
        Initializes the the value of weights and edges of each node 
        and also assign each node to their respective communities (each node is assigned a community as seen in the self candidate)

        we also compute the likelihood of each node in their original communities (they are all constant in this case)
        '''
        ## loop through the matrix

        for community, node in enumerate(self.corr_mat):

            #self.communities[community]=node
            self.weights[community]=1
            self.edges[community]=2
            ### initialize the likelihood for each community with a constant value
            # when the edges =2 and weight =1\
            self.likelihood[community]= 0.144
            self.candidates.append(community)
        #print("succesfully initialize algorithm....!!! the self likelihood is {}".format(self.likelihood))

    def is_candidate(self,node):

        '''
        Input  :   Integer (node) 
        output :   Boolean

                   return True if the candidate node is not yet merge (so we can merge!)
                   return False if the candidate has already been merged to a community
        '''
        ### verify its a candidate to merge
        if node in self.seen : return False
        else: return True

    def merge_communities(self,curr_community,new_community):

        '''
        Input:  curr_community (current community), new_community (new community)

                compute the weights and edges  when merging two communities using the likelihood formula
                we then compute the change in likelihood `delta L` (difference between likelihood of merge communities 
                and the sum of the individual communities)

        
        '''

        ## merging two communities
        self.weights[new_community] += 2*self.corr_mat[new_community][curr_community]+ self.corr_mat[curr_community][curr_community]
        self.edges[new_community]   += 2+ self.edges[curr_community]
        #print("succesfully merged two communites.....!!!")

    def unmerge_communities(self,curr_community,new_community):
        ## disentangle two communities

        self.weights[new_community]-= 2*self.corr_mat[new_community][curr_community]+self.corr_mat[curr_community][curr_community]
        self.edges[new_community]  -= 2+self.edges[curr_community]
       

        #print("succesfully unmeged two communities.....!!!")

    def Likelihood(self,community):

        '''
        Input  : Community 
        Output : The likelihood of that community
        '''
        ### computing the likelihood using the Giarda-Marsili likelihood function
        output= np.log(self.edges[community]/self.weights[community])-(self.edges[community]-1)*np.log((self.edges[community]**2-self.edges[community])/(self.edges[community]**2-self.weights[community]))
        return 0.5*output


    def Alc(self):

        '''
        The ALC algorithm goes as follows:

          Randomly merge a community with the remaining communities and compute the change in Likelihood
          Select the communities whose change in likelihood  is the smallest (merge the communities)
          Iterate this process until no further clustering can be done
        '''
        ##### performing the clustering

        while len(self.seen)<len(self.candidates):
            ## stopping criteria

            index= random.randrange(len(self.candidates))
            candidate= self.candidates[index]

            if not self.is_candidate(candidate):
                continue
            else:
                for node in self.candidates:
                    if (candidate== node): continue
                    #elif node in self.seen: continue
                    else:
                        self.seen.add(candidate)
                        ### compute the delta,likelihood
                        l_candidate = self.likelihood[candidate]
                        l_node      = self.likelihood[node]

                        #print("likelihood candidate {}".format(l_candidate))
                        
                        ## start by merging two candidates (communities)
                        self.merge_communities(candidate,node)
                        ### using the function in paper
                        delta_LC= self.Likelihood(node)- (l_candidate+l_node)
                        self.keys[node]=delta_LC
                        print("likelihood node {} is {}, likelihood of the candidate {} is {}, delta likelihood {}".format(node,l_node, candidate,l_candidate,delta_LC))
                        #print("change in likelihood {}".format(delta_LC))
                        ### unmerge communities 
                        self.unmerge_communities(candidate,node)

                output= [idx for idx,value in self.keys.items() if (value<=0.967)and value>=0.95]
                print(output)
                ### now we verify if they candidate can be merged 
                for idx in output:
                    ### merge communities
                    #self.merge_communities(candidate,self.curr_node)
                    self.result[candidate][idx]=self.corr_mat[candidate][idx]
                    print("merge node {} and node {}".format(candidate,idx))
                self.keys=defaultdict()
                self.curr_node=None
                self.delta_lc=-float('inf')

        return self.result
