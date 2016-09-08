"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from numpy import dot, hstack, vstack, ones
from par_tsne import *
from scipy.io import loadmat
from scipy.special import expit

train_mnist = loadmat('mnist_train.mat')
test_mnist = loadmat('mnist_test.mat')
layers = [500, 500, 2000, 2]
network = train_par_tsne(
    train_mnist['train_X'],
    train_mnist['train_labels'],
    layers,
    "CD1")

# map the test data
n = len(network)
mapped_X = hstack((test_mnist['test_X'], ones((len(test_mnist['test_X']), 1))))
for k in range(n - 1):
    mapped_X = expit(dot(mapped_X, vstack(
        (network[k]['W'], network[k]['bias_upW']))))
    mapped_X = hstack((mapped_X, ones((len(mapped_X), 1))))
mapped_X = dot(mapped_X, vstack((network[-1]['W'], network[-1]['bias_upW'])))

# plot the mapped test data
import matplotlib.pyplot as plt

plt.scatter(mapped_X[:, 0], mapped_X[:, 1], c=test_mnist['test_labels'])
plt.show()
