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

# np.random.rand = uniform random generation, [0, 1]
# np.random.randn = normal random generation, N(0, 1), 정규분포 난수
r1 = 10*(np.random.rand(2,2) - 0.5) # [-1, 1]
r2 = 10*(np.random.rand(2,2) - 0.5) # [-1, 1]
r3 = 10*(np.random.rand(2,2) - 0.5) # [-1, 1]

#target value
t1 = 50*(np.random.rand(2,1) - 0.5)
t2 = 50*(np.random.rand(2,1) - 0.5)
t3 = 50*(np.random.rand(2,1) - 0.5)

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

# unlabel data
x = np.hstack((x1, x2, x3))

plt.figure(figsize = (5,5))
plt.plot(x[0,:],x[1,:],".",markersize = 1, color = [0.75, 0.75, 0.75])
# +
D = 2
K = 3
N = 2000

# mu를 랜덤으로 초기화, 각 열 = 각 cluster의 중심
mu = 50*(np.random.rand(D,K) - 0.5)

# n번째 sample부터, k번째 cluster의 중심까지의 거리**2
dist2 = np.zeros((N,K)) 

for iter in range(0,10):
    for k in range(0,K):
        for n in range(0,N):
            dist2[n,k] = np.sum((x[:,n] - mu[:,k])**2)

    for k in range(0,K):
        dist2[:,k] = np.sum((x - np.matlib.repmat(mu[:,k].reshape(2,1),1,N))**2, axis = 0)

    # 어느 cluster까지 제일 가까운지를 c에 저장
    c = np.argmin(dist2, axis = 1) # dist = (N x K), 

    plt.figure(figsize = (9,3))
    plt.subplot(1,3,1)
    plt.plot(x[0,:],x[1,:],".",markersize = 1, color = [0.75, 0.75, 0.75])
    plt.plot(mu[0,0],mu[1,0],"C0x",markersize = 10)
    plt.plot(mu[0,1],mu[1,1],"C1x",markersize = 10)
    plt.plot(mu[0,2],mu[1,2],"C2x",markersize = 10)

    plt.subplot(1,3,2)
    plt.plot(x[0,c == 0],x[1,c == 0],"C0.",markersize = 1)
    plt.plot(x[0,c == 1],x[1,c == 1],"C1.",markersize = 1)
    plt.plot(x[0,c == 2],x[1,c == 2],"C2.",markersize = 1)

    plt.subplot(1,3,3)
    plt.plot(x[0,:],x[1,:],".",markersize = 1, color = [0.75, 0.75, 0.75])
    plt.plot(mu[0,0],mu[1,0],"C0x",markersize = 10)
    plt.plot(mu[0,1],mu[1,1],"C1x",markersize = 10)
    plt.plot(mu[0,2],mu[1,2],"C2x",markersize = 10)

    # 각 cluster에 대해 mu를 업데이트
    for k in range(0,K):
        mu[:,k] = np.mean(x[:,c == k], axis = 1)

    plt.plot(mu[0,0],mu[1,0],"C0+",markersize = 10)
    plt.plot(mu[0,1],mu[1,1],"C1+",markersize = 10)
    plt.plot(mu[0,2],mu[1,2],"C2+",markersize = 10)
# -

a = np.array([[1,2,3],[4,5,6]])
print(a)
print(a**2)
print(np.sum(a))
print(np.sum(a, axis = 0)) # 0번째 index = row에 대해서 더해라 = 세로방향
print(np.sum(a, axis = 1)) # 1번째 index = col에 대해서 더해라 = 가로방향

a = np.array([[1,2],[3,4]])
print(a)
print(np.matlib.repmat(a,2,3))
