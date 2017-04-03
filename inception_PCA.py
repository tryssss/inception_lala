"""Extract main features in training data and evaluate data by using compact PCA
	
   -----def _compact_pca(X)
   --reture V, S, mean_X
   --Args:
	X: feature matrix, [m,n] means that m objects and each object has n features, in 	which n>>m;
	V: projection matrix;
	S: variance matrix;
	mean_x: mean value of X;
   
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np

import tensorflow as tf

import math

FLAGS = tf.app.flags.FLAGS


def _pca(X_list):
    """
    Principal Component Analysis
    input: X, matrix with training data stored as flattened arrays in rows
    return: projection matrix (with important dimensions first), variance
    and mean.
    """
    # get dimensions
    X = np.array(X_list)
    num_data, dim = X.shape
    # average
    mean_X = X.mean(axis=0)
    avgs = np.tile(mean_X, (num_data, 1))
    X = X - avgs    
    # PCA - compact trick used
    M = X.dot(X.T)          # covariance matrix, AA', not the A'A like usual
    e,EV = np.linalg.eig(M) # compute eigenvalues and eigenvectors
    tmp = ((X.T).dot(EV)).T # this is the compact trick
    V = tmp
    # sort
    e = _make_negative2zero(e, num_data)# let e be zero will not effect the result, normally
    S = np.sqrt(e)
    e_nd = np.argsort(S)
    # cut off
    e_nd = e_nd[::-1]
    #e_nd = e_nd[0:k]
    V = V[e_nd, :]
    S = S[e_nd]
    for i in range(V.shape[1]):
      V[:,i] /= S
    # obtain k first terms of V and average
    avgs = avgs[e_nd, :]
    #return the projection matrix and the average of X_class, each line is the same
    return V, avgs[1, :]

def compute_project(V, X, avgs, k):
    V = V[0:k, :]
    projections = []
    for i in range(X.shape[0]):
       projections.append(V.dot((X[i,:] - avgs).T))
    return projections


def _find_max_autocorrelation(X_list, num_sample, MIN_k, MAX_k):
    # compute max coefficient of autocorrelation and k in class X_list
    # return k of maxcoeff and max coefficient
    V, mean_X = _pca(X_list)       # compute pca first
    X = np.array(X_list)
    MAX_coeff = 0                 # max coefficient
    MAX_coeff_k = MIN_k           # k of MAX_coeff
    coeff_auto = np.empty((num_sample,num_sample)) # create a array restored coeff,31*(max-min+1)
    while MIN_k < MAX_k:          # find max coefficient in every k in every classes
      projections = compute_project(V, X, mean_X, MIN_k)
      for j in range(num_sample):
        for k in range(num_sample):
          coeff_auto[j][k] = _correlation(projections[j], projections[k]) # coeff_auto[0][1] means that the coeff between sample 0 and sample 1
      #print(coeff_auto)
      sum_coeff_auto = coeff_auto.sum()
      #print(sum_coeff_auto)
      if MAX_coeff < sum_coeff_auto:
         MAX_coeff = sum_coeff_auto
         MAX_coeff_k = MIN_k
      MIN_k += 1

    #print(MIN_k)

    mean_coeff = MAX_coeff/((num_sample)*(num_sample))
    return MAX_coeff_k, mean_coeff, coeff_auto, X
"""
def _compute_cross_correlation(X_list, num_class, MIN_k, MAX_k):
    # compute max cross coefficient of autocorrelation and k in classes X_list
    # return k of maxcoeff and max coefficient
    X = []
    for i in range(num_class):
      X = [X, (_pca(X_list[i]))[i,:]]       # compute pca first, random choose sample
    X = np.array(X)
    MIN_coeff = 0                 # Init  coefficient
    MIN_coeff_k = MIN_k           # k of MAX_coeff
    coeff_auto = np.empty((num_class, num_class)) # create a array restored coeff,31*(max-min+1)
    while MIN_k < MAX_k:          # find max coefficient in every k in every classes
      for j in range(num_class):
        for k in range(num_class):
          coeff_auto[j][k] = _correlation(X[j,0:MIN_k],X[k,0:MIN_k]) # coeff_auto[0][1] means that the coeff between sample 0 and sample 1
      #print(coeff_auto)
      sum_coeff_auto = coeff_auto.sum()
      #print(sum_coeff_auto)
      if MIN_coeff < sum_coeff_auto:
         MIN_coeff = sum_coeff_auto
         MIN_coeff_k = MIN_k
      MIN_k += 1

    #print(MIN_k)

    mean_coeff = MIN_coeff/((num_sample)*(num_sample))
    return MIN_coeff_k, mean_coeff, coeff_auto, X
"""

def _correlation(a, b):
    # data centerize
    N = a.shape
    a = a-a.sum()/N
    b = b-b.sum()/N
    # compute a.*b
    ab = a.dot(b.T)
    # compute the module of vector
    a_m = np.linalg.norm(a)
    b_m = np.linalg.norm(b)
    # compute correlation between a and b
    #std_a = np.std(a)
    #std_b = np.std(b)
    #cov_ab = np.cov(a,b)
    #row, cloumn = cov_ab.shape
    #coeff = cov_ab.sum()/(std_a * std_b)
    coeff = ab/(a_m*b_m)
    return np.abs(coeff)

def _make_negative2zero(X, k):
   i = 0
   while i < k:
      if X[i] < 0:
         X[i] = 0
      i += 1
   return X

def _find_max_value(X):
    # input X is a array
    # return location in the array
    row, column = X.shape
    N = X.argmax()
    return math.floor((N/row)-1),(N % column)-1
