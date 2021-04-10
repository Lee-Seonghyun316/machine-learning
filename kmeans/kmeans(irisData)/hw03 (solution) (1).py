# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from sklearn.datasets import load_iris
import numpy as np 
import numpy.matlib 
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import multivariate_normal

iris = load_iris()
x = iris.data.T
D,N = x.shape

# +
# 사용자 설정, hyperparameter
K = 3
maxIter = 10
mu0 = np.array([[4.0, 5.0, 6.0],[4.0, 2.0, 3.0],[1.0, 2.0, 3.0],[0.2, 1.0, 2.0]])
#mu0 = np.random.rand(D,K)

plt.figure(figsize = (8,8))
for i in range(0,D):
    for j in range(0,D):
        plt.subplot(D,D,i*D + j + 1)
        plt.plot(x[i,:],x[j,:],".",markersize = 3,color = [0.75,0.75,0.75])
        for k in range(0,K):
            plt.plot(mu0[i,k],mu0[j,k],'C%do--'%k)
        plt.xlabel('%d-dim'%(i+1))
        plt.ylabel('%d-dim'%(j+1))
        plt.tight_layout()

# +
# 사용자 설정, hyperparameter
K = 3
maxIter = 10
    
mu = np.copy(mu0)

# initialize
muTrail = np.empty((D,K,maxIter+1))
muTrail[:] = np.nan
muTrail[:,:,0] = mu
        
# k-means clustering
dist2 = np.zeros((N,K)) 
for iter in range(0,maxIter):
    # Estep
    for k in range(0,K):
        dist2[:,k] = np.sum( (x - np.matlib.repmat(mu[:,k].reshape(D,1),1,N))**2, axis = 0)
    c = np.argmin(dist2, axis = 1)

    # Mstep
    for k in range(0,K):
        mu[:,k] = np.mean(x[:,c == k], axis = 1)
        
    # save
    muTrail[:,:,iter+1] = mu

plt.figure(figsize = (8,8))
for i in range(0,D):
    for j in range(0,D):
        plt.subplot(D,D,i*D + j + 1)
        plt.plot(x[i,:],x[j,:],".",markersize = 3,color = [0.75,0.75,0.75])
        for k in range(0,K):
            plt.plot(muTrail[i,k],muTrail[j,k],'C%dx--'%k)
            plt.plot(mu0[i,k],mu0[j,k],'C%do--'%k)
        plt.xlabel('%d-dim'%(i+1))
        plt.ylabel('%d-dim'%(j+1))
        plt.tight_layout()
plt.savefig('iris-kmeans.eps')

# +
# 사용자 설정, hyperparameter
K = 3
maxIter = 10

# 초기 변수 설정
pi = np.ones((1,K))
pi = pi / np.sum(pi)

sigma = np.zeros((D,D,K)) # identity, I 행렬로 초기화
for k in range(0,K):
    sigma[:,:,k] = np.identity(D)
    
mu = np.copy(mu0)
print(mu)

# initialize
muTrail = np.empty((D,K,maxIter+1))
muTrail[:] = np.nan
muTrail[:,:,0] = mu

# MoG clustering
for iter in range(0,maxIter):
    # Estep
    gamma = np.zeros((K,N))
    for k in range(0,K):
        gamma[k,:] = pi[0,k]*multivariate_normal.pdf(x.T,mean = mu[:,k], cov = sigma[:,:,k])
    gamma = gamma / np.sum(gamma,axis=0)

    # Mstep
    Num = np.zeros((1,K))
    res = np.zeros((D,N))
    gamma_res = np.zeros((D,N))
    sigma = np.zeros((D,D,K))
    for k in range(0,K):
        Num[0,k] = np.sum(gamma[k,:])
        mu[:,k] = np.sum(np.matlib.repmat(gamma[k,:], D, 1)*x, axis=1) / Num[0,k]
        res = (x - np.matlib.repmat(mu[:,k].reshape(D,-1),1,N))
        gamma_res = np.matlib.repmat(gamma[k,:].reshape(1,-1),D,1) * res
        for n in range(0,N):
            sigma[:,:,k] += res[:,n].reshape(-1,1).dot(gamma_res[:,n].reshape(-1,1).T)
        sigma[:,:,k] /= Num[0,k]

        pi[0,k] = Num[0,k] / N    
        
    # save
    muTrail[:,:,iter+1] = mu

plt.figure(figsize = (8,8))
for i in range(0,D):
    for j in range(0,D):
        plt.subplot(D,D,i*D + j + 1)
        plt.plot(x[i,:],x[j,:],".",markersize = 3,color = [0.75,0.75,0.75])
        for k in range(0,K):
            plt.plot(muTrail[i,k,:],muTrail[j,k,:],'C%dx--'%k)
            plt.plot(mu0[i,k],mu0[j,k],'C%do--'%k)
        plt.xlabel('%d-dim'%(i+1))
        plt.ylabel('%d-dim'%(j+1))
        plt.tight_layout()
plt.savefig('iris-mog.eps')

