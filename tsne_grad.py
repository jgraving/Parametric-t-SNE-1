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


def compute_Qvalues(x, v, n):

    sum_x = np.sum(x ** 2.0, axis=1)
    num = 1 + (sum_x[:, np.newaxis] + (sum_x.T - 2.0 * np.dot(x, x.T))) / v
    Q = num ** -((v + 1) / 2.0)
    np.fill_diagonal(Q, 0)  # np.arange(0,len(Q),n+1)
    Q /= np.sum(Q)
    Q[Q < np.finfo(float).eps] = np.finfo(float).eps
    num = 1 / num
    np.fill_diagonal(num, 0)

    return Q, num


def tsne_grad(x, X, P, network, v):

    n = len(X)
    no_layers = len(network)

    ii = 0
    for i in range(no_layers):
        network[i]["W"] = x[
            ii:ii +
            network[i]["W"].size].reshape(
            network[i]["W"].shape,
            order='F')
        ii += network[i]["W"].size
        network[i]["bias_upW"] = x[
            ii:ii +
            network[i]["bias_upW"].size].reshape(
            network[i]["bias_upW"].shape,
            order='F')
        ii += network[i]["bias_upW"].size

    activations = []
    activations.append(np.append(X, np.ones((len(X), 1)), axis=1))
    for i in range(no_layers - 1):
        act_tmp = expit(
            np.dot(
                activations[i], np.vstack(
                    (network[i]["W"], network[i]["bias_upW"]))))
        activations.append(
            np.append(
                act_tmp, np.ones(
                    (len(act_tmp), 1)), axis=1))
    activations.append(
        np.dot(
            activations[
                no_layers -
                1],
            np.vstack(
                (network[
                    no_layers -
                    1]["W"],
                    network[
                    no_layers -
                    1]["bias_upW"]))))

    Q, num = compute_Qvalues(activations[-1], v, len(X))
    C = np.sum(P * np.log((P + np.finfo(float).eps) / (Q + np.finfo(float).eps)))

    Ix = np.zeros(activations[-1].shape)
    stiffness = 4 * ((v + 1) / (2 * v)) * (P - Q) * num
    for i in range(n):
        Ix[i] = np.sum((activations[-1][i] - activations[-1])
                       * stiffness[:, i, np.newaxis], axis=0)

    dW = no_layers * [0]
    db = no_layers * [0]
    for i in range(no_layers - 1, -1, -1):
        delta = np.dot(activations[i].T, Ix)
        dW[i] = delta[:-1]
        db[i] = delta[-1]

        if i > 0:
            Ix = np.dot(Ix, np.vstack((network[i]['W'], network[i]['bias_upW'])).T) * \
                activations[i] * (1 - activations[i])
            Ix = Ix[:, :-1]

    dC = []
    ii = 0
    for i in range(no_layers):
        dC = np.append(dC, dW[i].reshape(-1, order='F'))
        ii = ii + dW[i].size
        dC = np.append(dC, db[i].reshape(-1, order='F'))
        ii = ii + db[i].size

    return (C, dC)
