# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 12:25:51 2016

@author: paolo
"""

import profile
import pstats
from rbm import train_rbm

from scipy.io import loadmat

train_mnist = loadmat('/home/paolo/source-test/parametric_tsne/mnist_test.mat')
train_X = train_mnist['test_X']
train_labels = train_mnist['test_labels']
layers=[250,50,2]

# Create 5 set of stats
filenames = []
for i in range(5):
    filename = 'profile_stats_%d.stats' % i
    profile.run('print %d, train_rbm(train_X,layers[0],0.01,3)' % i, filename)

# Read all 5 stats files into a single object
stats = pstats.Stats('profile_stats_0.stats')
for i in range(1, 5):
    stats.add('profile_stats_%d.stats' % i)

# Clean up filenames for the report
stats.strip_dirs()

# Sort the statistics by the cumulative time spent in the function
stats.sort_stats('cumulative')

stats.print_stats()

# Read all 5 stats files into a single object
stats = pstats.Stats('profile_stats_0.stats')
for i in range(1, 5):
    stats.add('profile_stats_%d.stats' % i)
stats.strip_dirs()
stats.sort_stats('cumulative')

# limit output to lines with "(fib" in them
stats.print_stats('\(trai')
