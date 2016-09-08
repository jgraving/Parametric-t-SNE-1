# -*- coding: utf-8 -*-
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

@author: paolo
"""

import numpy as np
from scipy.special import expit


def compute_recon_err(machine, X):

    if isinstance(machine, list):

        err = np.zeros(len(machine))
        vis = X
        for i in range(len(machine)):
            if i < len(machine):
                hid = expit(
                    np.dot(
                        vis,
                        machine[i]['W']) +
                    machine[i]['bias_upW'])
            else:
                hid = vis * machine[i]['W'] + machine[i]['bias_upW']

        rec = expit(np.dot(hid, machine[i]['W'].T) + machine[i]['bias_downW'])
        err[i] = np.sum((vis - rec)**2) / len(X)
        vis = hid

    else:

        hid = expit(np.dot(X, machine['W']) + machine['bias_upW'])
        rec = expit(np.dot(hid, machine['W'].T) + machine['bias_downW'])
        err = np.sum((X - rec)**2) / len(X)

    return err


def train_rbm(X, h=30, eta=0.1, max_iter=30, weight_cost=0.0002):

    initial_momentum = 0.5
    final_momentum = 0.9

    (n, v) = X.shape
    batch_size = 100
    machine = {'W': np.random.randn(v, h) * 0.1,
               'bias_upW': np.zeros(h),
               'bias_downW': np.zeros(v)
               }
    delta_W = np.zeros((v, h))
    delta_bias_upW = np.zeros(h)
    delta_bias_downW = np.zeros(v)

    # Main loop
    train_err = np.zeros(max_iter)
    for i in range(max_iter):

        ind = np.random.permutation(n)

        if i < 5:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        for batch in np.arange(0, n, batch_size):
            if batch + batch_size <= n:

                vis1 = X[ind[batch:np.min([batch + batch_size, n])]]

                hid1 = expit(np.dot(vis1, machine['W']) + machine['bias_upW'])

                hid_states = np.asarray(
                    hid1 > np.random.rand(
                        hid1.shape[0],
                        hid1.shape[1]),
                    dtype=float)

                vis2 = expit(
                    np.dot(
                        hid_states,
                        machine['W'].T) +
                    machine['bias_downW'])

                hid2 = expit(np.dot(vis2, machine['W']) + machine['bias_upW'])

                posprods = np.dot(vis1.T, hid1)
                negprods = np.dot(vis2.T, hid2)
                delta_W = momentum * delta_W + eta * \
                    ((posprods - negprods) / batch_size -
                     weight_cost * machine['W'])
                delta_bias_upW = momentum * delta_bias_upW + eta / \
                    batch_size * (np.sum(hid1, axis=0) - np.sum(hid2, axis=0))
                delta_bias_downW = momentum * delta_bias_downW + eta / \
                    batch_size * (np.sum(vis1, axis=0) - np.sum(vis2, axis=0))

                machine['W'] += delta_W
                machine['bias_upW'] += delta_bias_upW
                machine['bias_downW'] += delta_bias_downW

        err_tmp = compute_recon_err(machine, X)
        print "Iteration %d (train rec. error = %f)" % (i, err_tmp)
        train_err[i] = err_tmp

    print " "
    err = compute_recon_err(machine, X)

    return machine, err


def train_linear_rbm(X, h=20, eta=0.001, max_iter=50, weight_cost=0.0002):

    initial_momentum = 0.5
    final_momentum = 0.9

    (n, v) = X.shape
    batch_size = 100
    machine = {'W': np.random.randn(v, h) * 0.1,
               'bias_upW': np.zeros(h),
               'bias_downW': np.zeros(v)
               }
    delta_W = np.zeros((v, h))
    delta_bias_upW = np.zeros(h)
    delta_bias_downW = np.zeros(v)

    # Main loop
    for i in range(max_iter):

        err = 0
        ind = np.random.permutation(n)
        if i < 5:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        for batch in np.arange(0, n, batch_size):
            if batch + batch_size <= n:

                vis1 = X[ind[batch:np.min([batch + batch_size, n])]]

                hid1 = np.dot(vis1, machine['W']) + machine['bias_upW']

                hid_states = hid1 + \
                    np.random.randn(hid1.shape[0], hid1.shape[1])

                vis2 = expit(
                    np.dot(
                        hid_states,
                        machine['W'].T) +
                    machine['bias_downW'])

                hid2 = np.dot(vis2, machine['W']) + machine['bias_upW']

                posprods = np.dot(vis1.T, hid1)
                negprods = np.dot(vis2.T, hid2)
                delta_W = momentum * delta_W + eta * \
                    ((posprods - negprods) / batch_size -
                     weight_cost * machine['W'])
                delta_bias_upW = momentum * delta_bias_upW + eta / \
                    batch_size * (np.sum(hid1, axis=0) - np.sum(hid2, axis=0))
                delta_bias_downW = momentum * delta_bias_downW + eta / \
                    batch_size * (np.sum(vis1, axis=0) - np.sum(vis2, axis=0))

                machine['W'] += delta_W
                machine['bias_upW'] += delta_bias_upW
                machine['bias_downW'] += delta_bias_downW

                err += np.sum(np.sum((vis1 - vis2)**2, axis=0))

        print "Iteration %d (train rec. error ~ %f)" % (i, err / n)

    return machine


def train_rbm_pcd(X, h=20, eta=0.02, max_iter=100, weight_cost=0):

    initial_momentum = 0.9
    final_momentum = 0.9

    (n, v) = X.shape
    batch_size = 100
    machine = {'W': np.random.randn(v, h) * 0.1,
               'bias_upW': np.zeros(h),
               'bias_downW': np.zeros(v)
               }
    delta_W = np.zeros((v, h))
    delta_bias_upW = np.zeros(h)
    delta_bias_downW = np.zeros(v)

    # Initialize Markov chain
    hid2 = np.array(np.random.rand(batch_size, h) > 0.5, dtype=float)

    for i in range(0, max_iter):

        ind = np.random.permutation(n)

        if i < 5:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        for batch in arange(0, n, batch_size):
            if batch + batch_size <= n:

                vis1 = X[ind[batch:np.min([batch + batch_size, n])]]

                hid1 = expit(np.dot(vis1, machine['W']) + machine['bias_upW'])

                # Sample state using the Markov chain
                hid_states = np.asarray(
                    hid2 > np.random.rand(
                        hid2.shape[0],
                        hid2.shape[1]),
                    dtype=float)

                vis2 = expit(
                    np.dot(
                        hid_states,
                        machine['W'].T) +
                    machine['bias_downW'])

                hid2 = expit(np.dot(vis2, machine['W']) + machine['bias_upW'])

                posprods = np.dot(vis1.T, hid1)
                negprods = np.dot(vis2.T, hid2)
                delta_W = momentum * delta_W + eta * \
                    ((posprods - negprods) / batch_size -
                     weight_cost * machine['W'])
                delta_bias_upW = momentum * delta_bias_upW + eta / \
                    batch_size * (np.sum(hid1, axis=0) - np.sum(hid2, axis=0))
                delta_bias_downW = momentum * delta_bias_downW + eta / \
                    batch_size * (np.sum(vis1, axis=0) - np.sum(vis2, axis=0))

                machine['W'] += delta_W
                machine['bias_upW'] += delta_bias_upW
                machine['bias_downW'] += delta_bias_downW

        err = compute_recon_err(machine, X[0:5000])
        print "Iteration %d (train rec. error = %f)" % (i, err_tmp)

    return machine
