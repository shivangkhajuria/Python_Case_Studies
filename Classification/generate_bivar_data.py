

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
