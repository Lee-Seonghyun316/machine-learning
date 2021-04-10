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
import numpy as np 
import numpy.matlib 
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import multivariate_normal

# np.random.rand = uniform random generation, [0, 1]
# np.random.randn = normal random generation, N(0, 1)
r1 = 10*(np.random.rand(2,2) - 0.5) # [-1, 1]
r2 = 10*(np.random.rand(2,2) - 0.5) # [-1, 1]
r3 = 10*(np.random.rand(2,2) - 0.5) # [-1, 1]

#t1 = 50*(np.random.rand(2,1) - 0.5)
#t2 = 50*(np.random.rand(2,1) - 0.5)
#t3 = 50*(np.random.rand(2,1) - 0.5)
t1 = np.array([10,10]).reshape(2,1)
t2 = np.array([10,-10]).reshape(2,1)
t3 = np.array([-10,-10]).reshape(2,1)

x1 = np.random.randn(2,1000)
x1 = r1.dot(x1) + t1

x2 = np.random.randn(2,500)
x2 = r2.dot(x2) + t2

x3 = np.random.randn(2,500)
x3 = r3.dot(x3) + t3

plt.figure(figsize = (5,5))
plt.plot(x1[0,:],x1[1,:],"C0.",markersize = 1)
plt.plot(x2[0,:],x2[1,:],"C1.",markersize = 1)
plt.plot(x3[0,:],x3[1,:],"C2.",markersize = 1)

x = np.hstack((x1, x2, x3))
plt.figure(figsize = (5,5))
plt.plot(x[0,:],x[1,:],".",markersize = 1,color = [0.75,0.75,0.75])
plt.xlim([np.min(x[0,:]),np.max(x[0,:])])
plt.ylim([np.min(x[1,:]),np.max(x[1,:])])

# +
K = 3
D,N = x.shape

# 초기화
pi = np.random.rand(1,K)
pi = pi / np.sum(pi)
print(pi)

mu = 20*(np.random.rand(2,K) - 0.5)
sigma = np.zeros((D,D,K)) # identity, I 행렬로 초기화
for k in range(0,K):
    sigma[:,:,k] = np.array([[1,0],[0,1]])

# 2-d Gaussian, mixture Gaussian
# contour = 등고선

u = np.linspace(-50,50,100)
v = np.linspace(-50,50,100)
uu, vv = np.meshgrid(u, v) # 바둑판 나누기
mog_pdf = np.zeros(uu.shape) # mog distribution을 계산을 해서 저장
for k in range(0,K):
    # 2-d Gaussian, pdf 평가 = multivariate_normal.pdf
    pdf = multivariate_normal.pdf(np.hstack((uu.reshape(-1,1),vv.reshape(-1,1))), mean = mu[:,k], cov = sigma[:,:,k])
    mog_pdf += pi[0,k] * pdf.reshape(100,100)

plt.figure(figsize = (5,5))
plt.plot(x[0,:],x[1,:],".",markersize = 1, color = [0.75, 0.75, 0.75])
plt.contour(uu,vv,mog_pdf)
plt.xlim([np.min(x[0,:]),np.max(x[0,:])])
plt.ylim([np.min(x[1,:]),np.max(x[1,:])])
# -

for iter in range(0,50):
    # Estep: gamma (responsibility) 구하기
    # 주어진 mu, sigma, pi에 대해서

    plt.figure(figsize = (10,5))
    plt.subplot(1,2,1)
    plt.plot(x[0,:],x[1,:],".",markersize = 1, color = [0.75, 0.75, 0.75])
    plt.contour(uu,vv,mog_pdf)
    plt.xlim([np.min(x[0,:]),np.max(x[0,:])])
    plt.ylim([np.min(x[1,:]),np.max(x[1,:])])

    # 회색 점들 = x, (DxN)
    # pdf = probability density function, 확률밀도함수
    gamma = np.zeros((K,N)) # k번째 cluster, n번째 sample
    for k in range(0,K):
        gamma[k,:] = pi[0,k]*multivariate_normal.pdf(x.T,mean = mu[:,k], cov = sigma[:,:,k])

    # gamma = K x N 행렬, 
    # normalize = 세로방향으로 더해서, 나눠야함
    gamma = gamma / np.sum(gamma,axis=0)


    # Mstep: mu, sigma, pi 구하기
    # 주어진 gamma에 대해서

    Num = np.zeros((1,K))
    res = np.zeros((D,N))
    gamma_res = np.zeros((D,N))
    sigma = np.zeros((D,D,K))
    for k in range(0,K):
        Num[0,k] = np.sum(gamma[k,:]) # N_k

        mu[:,k] = np.sum(np.matlib.repmat(gamma[k,:], D, 1)*x, axis=1) / Num[0,k]
        # mu = sum gamma_nk * x_n / sum gamma_nk = sum gamma_nk * x_n / N_k

        # residual = x - mu[:,k]
        # gamma*residual = gamma[k,:] * (x - mu[:,k])
        # sigma = gamma * (x - mu) * (x - mu)^T / N_k
        res = (x - np.matlib.repmat(mu[:,k].reshape(2,-1),1,N))
        gamma_res = np.matlib.repmat(gamma[k,:].reshape(1,-1),D,1) * res
        for n in range(0,N):
            sigma[:,:,k] += res[:,n].reshape(-1,1).dot(gamma_res[:,n].reshape(-1,1).T)
        sigma[:,:,k] /= Num[0,k]

        pi[0,k] = Num[0,k] / N

    # mog pdf를 새로 업데이트
    mog_pdf = np.zeros(uu.shape) # mog distribution을 계산을 해서 저장
    for k in range(0,K):
        # 2-d Gaussian, pdf 평가 = multivariate_normal.pdf
        pdf = multivariate_normal.pdf(np.hstack((uu.reshape(-1,1),vv.reshape(-1,1))), mean = mu[:,k], cov = sigma[:,:,k])
        mog_pdf += pi[0,k] * pdf.reshape(100,100)

    plt.subplot(1,2,2)
    plt.plot(x[0,:],x[1,:],".",markersize = 1, color = [0.75, 0.75, 0.75])
    plt.contour(uu,vv,mog_pdf)
    plt.xlim([np.min(x[0,:]),np.max(x[0,:])])
    plt.ylim([np.min(x[1,:]),np.max(x[1,:])])
    

Num = np.zeros((1,K))
for k in range(0,K):
    print(Num[k])
