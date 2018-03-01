# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:54:02 2018

@author: The Capricorn
"""

import numpy as np

def distance(p1,p2):
        
    return np.sqrt(np.sum(np.power(p2-p1,2)))

p1 = np.array([1,2])
p2 = np.array([5,6])

distance(p1,p2)

import random

def majority_vote(votes):
    """
    Returns the most common occuring vote count
    """
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
    winners = []
    for vote in vote_counts:
        if vote_counts[vote] == max(vote_counts.values()):
            winners.append(vote)
    
    return random.choice(winners)

import scipy.stats as ss
    
def majority_vote_short(votes):
    """
    Returns the most common occuring vote count
    """
    mode, count = ss.mstats.mode(votes)
            
    return mode



votes = [1,2,3,4,5,1,2,3,1,2,3,1,2,3,1,4,23,4,3,4,5,4,3,1,2,3,21,3,4,2,1,2,1,2]

vote_count = majority_vote(votes)


# loop over all the points
    # find distance of all points from the point
# sort the distances


import matplotlib.pyplot as plt
points = np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]])
p = np.array([2.5,2])



plt.axis([0,4,0,4])
plt.plot(points[:,0],points[:,1],"bo")
plt.plot(p[0],p[1],"ro")

def find_nearest_neighbors(p,points,k=3):
    """
    Finds the k nearest neighbors of point p and returns their indices
    """
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p,points[i])
    ind = distances.argsort()
    return ind[:k]

    
def knn_predict(p,points,outcomes,k = 3):
    """
    Finds the classification of point p among the outcomes and returns it.
    """
    ind = find_nearest_neighbors(p,points,k)
    return majority_vote(outcomes[ind])

outcomes = np.array([0,0,0,1,1,1,2,2,2])







































