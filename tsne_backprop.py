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

@author: paoloinglese
"""

from minimize_1 import *
import numpy as np


def Hbeta(D, beta):

    P = np.exp(-D * beta)
    sumP = np.sum(P, axis=0)
    H = np.log(sumP) + beta * np.sum(D * P, axis=0) / sumP
    P = P / sumP
    return H, P


def pdist_python(x):
    delta = sum(x.T**2)
    return delta[:, None] + delta[None, :] - 2 * np.dot(x, x.T)


def compute_gaussian_perplexity(x, u=15.0, tol=1e-4):

    n = len(x)
    P = np.zeros((n, n))
    beta = np.ones(n)
    logU = np.log(u)

    print "Computing pairwise distances..."
    DD = pdist_python(x)

    print "Computing P-values..."
    for i in range(n):

        min_beta = -np.inf
        max_beta = np.inf

        if i % 500 == 0:
            print "Computed P-values {} of {} datapoints...".format(i, n)

        Di = DD[i, np.arange(len(DD)) != i]
        H, thisP = Hbeta(Di, beta[i])

        Hdiff = H - logU
        tries = 0

        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                min_beta = beta[i]
                if np.isinf(max_beta):
                    beta[i] *= 2.0
                else:
                    beta[i] = (beta[i] + max_beta) / 2.0
            else:
                max_beta = beta[i]
                if np.isinf(min_beta):
                    beta[i] /= 2.0
                else:
                    beta[i] = (beta[i] + min_beta) / 2.0

            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        P[i, np.arange(len(P)) != i] = thisP

    print "Mean value of sigma: %f" % np.mean(np.sqrt(np.reciprocal(beta)))
    print "Minimum value of sigma: %f" % np.min(np.sqrt(np.reciprocal(beta)))
    print "Maximum value of sigma: %f" % np.max(np.sqrt(np.reciprocal(beta)))

    return P, beta


def tsne_backprop(network, train_X, train_labels,
                  max_iter, perplexity=30, v=1):

    n = len(train_X)
    batch_size = np.min([5000, n])
    ind = np.random.permutation(n)
    err = np.zeros(max_iter)

    print "Precomputing P-values..."
    curX = []
    P = []
    for batch in np.arange(0, n, batch_size):
        if batch + batch_size <= n:
            curX.append(
                (train_X[ind[batch:np.min([batch + batch_size, n])]]).astype(float))
            Ptmp, _ = compute_gaussian_perplexity(curX[-1], perplexity, 1e-5)
            Ptmp[np.isnan(Ptmp)] = 0
            Ptmp = (Ptmp + Ptmp.T) / 2.0
            Ptmp = Ptmp / np.sum(Ptmp)
            Ptmp[Ptmp < np.finfo(float).eps] = np.finfo(float).eps
            P.append(Ptmp)

    for it in range(max_iter):
        print "Iteration {}".format(it)
        b = 0
        for batch in np.arange(0, n, batch_size):
            if batch + batch_size <= n:

                x0 = []
                for i in range(len(network)):
                    x0 = np.append(x0, np.append(network[i]['W'].reshape(-1, order='F'),
                                                 network[i]['bias_upW'].reshape(-1, order='F')))

                xmin, _, _ = minimize(
                    x0, tsne_grad, 3, curX[b], P[b], network, v)

                b += 1

                # store new solution
                ii = 0
                for i in range(len(network)):
                    network[i]['W'] = xmin[
                        ii:ii +
                        network[i]['W'].size].reshape(
                        network[i]['W'].shape,
                        order='F')
                    ii = ii + network[i]['W'].size
                    network[i]['bias_upW'] = xmin[
                        ii:ii +
                        network[i]['bias_upW'].size].reshape(
                        network[i]['bias_upW'].shape,
                        order='F')
                    ii = ii + network[i]['bias_upW'].size

    return network
