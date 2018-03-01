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



import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt


def gen_syn_data(n=50):
    """Return two set of points from bivarite distribution of data"""
    points = np.concatenate((ss.norm(0,1).rvs((n,2)),ss.norm(1,1).rvs((n,2))),axis = 0)
    outcomes = np.concatenate(((np.repeat(0,n)), np.repeat(1,n)))
    return (points,outcomes)


   # plotting data
'''
n=20
plt.figure()
plt.plot(points[:n,0],points[:n,1],"go")
plt.plot(points[n:,0],points[n:,1],"ro")
plt.savefig('bivardata.pdf')
'''

def make_prediction_grid(limits,predictors,outcomes,k,h):
    """
    Classify each point on the prediction grid.
    """
    (x_min,x_max,y_min,y_max) = limits
    xs = np.arange(x_min,x_max,h)
    ys = np.arange(y_min,y_max,h)
    (xx,yy) = np.meshgrid(xs,ys)
    
    prediction_grid = np.zeros(xx.shape,dtype = int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict(p,predictors,outcomes,k)
            
    return (xs,ys,prediction_grid)
    

def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)


# Combining knn() func with synthetic data and make_prediction_grid()
    

(predictors,outcomes) = gen_syn_data()

k=5; filename="knn_synth_5.pdf"; limits=(-3,4,-3,4); h=0.1
(xx,yy,prediction_grid) = make_prediction_grid(limits,predictors,outcomes,k,h)
plot_prediction_grid(xx,yy,prediction_grid,filename)

k=50; filename="knn_synth_50.pdf"; limits=(-3,4,-3,4); h=0.1
(xx,yy,prediction_grid) = make_prediction_grid(limits,predictors,outcomes,k,h)
plot_prediction_grid(xx,yy,prediction_grid,filename)

































