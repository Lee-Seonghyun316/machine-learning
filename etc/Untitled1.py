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

r1 = 2*(np.random.rand(2,2)-0.5)#[-1,1]
r2 = 2*(np.random.rand(2,2)-0.5)#[-1,1]
r2 = 2*(np.random.rand(2,2)-0.5)#[-1,1]

t1 = 20*(np.random.rand(2,1)- 0.5)
t2 = 20*(np.random.rand(2,1)- 0.5)
t3 = 20*(np.random.rand(2,1)- 0.5)

#np.random.rand = uniform random generation [0,1]
#np.random.randn = normal random generation N(0,1)->0에서 1로 범위가 특정되지 않으나 많은 확률로 그렇게 나옴 clustering이 더 말이된다. 
x1 = np.random.randn(2,1000)
x1 = r1.dot(x1) + t1 #matrix 곱

x2 = np.random.randn(2,500)
x2 = r1.dot(x2) + t2 #matrix 곱

x3 = np.random.randn(2,500)
x3 = r1.dot(x3) + t3 #matrix 곱

plt.figure(figsize = (9,3))
#plt.subplot(1,3,1)#겹쳐서 그려지지 않도록
plt.plot(x1[0,:],x1[1,:],"C0.",markersize = 1)#포인츠에 .찍으로고, 색 다르게 C0
plt.plot(x2[0,:],x2[1,:],"C1.",markersize = 1)
plt.plot(x3[0,:],x3[1,:],"C2.",markersize = 1)

#unlabel data
x = np.hstack((x1 ,x2 ,x3))

plt.figure(figsize = (9,3))
plt.plot(x[0,:],x[1,:],".",markersize = 1, color = [0.75,0.75,0.75])

# +
K = 3

# 초기화 

pi = np.random.rand(1, 3)
pi = pi / np.sum(pi)
print(pi)

mu = 20*(np.random.rand(2, K) - 0.5)
sigma = np.zeros((2, 2, K))
for k in range(0,K):
    sigma[:, :, k] = np.array([[1, 0], [0, 1]])
    
u = np.linspace(-15, 15, 100)
v = np.linspace(-25, 25, 100)


# +
D = 2
K = 3 # 사용자가 설정하기 나름
N = 2000

#mu를 랜덤으로 초기화, 각 열 = 각 cluster의 중심
mu = 50*(np.random.rand(D,K)-0.5)

#N번째 셈픍부처 K번째 cluster중심까지의 거리 제곱
dist2 = np.zeros((N,K)

for k in range(0,K):
    for n in range(0,N):
        dist2[n,k] = np.sum((x[:,n] - mu[:,k])**2)
                 
# for k in range(0,K):
#     for n in range(0,N):
#         #dist2[n,k] = np.sum(x[:,n] - mu[:,k])**2
#         dist2[n,k] = np.sum(x[:,n]-mu[:,k])

# for k in range(0,K):
#     dist2[:,k] = np.sum(x - np.matlib.repmat(mu[:,k],reshape(2,1),1,N))**2
#      #K번째 cluster중심, D*N
                
#어느 cluster까지 제일 가까운지 c 에 저장
c = np.argmin(dist2,axis = 1)
                
plt.figure(figsize = (9,3))
plt.subplot(1,3,1)#겹쳐서 그려지지 않도록
plt.plot(x[0,:],x[1,:],".",markersize = 1, color = [0.75,0.75,0.75],axis = 0)
plt.plot(mu[0,0],mu[1,0],"C0x",markersize = 10)#포인츠에 .찍으로고, 색 다르게 C0
plt.plot(mu[0,1],mu[1,1],"C1x",markersize = 10)
plt.plot(mu[0,2],mu[1,1],"C2x",markersize = 10)
# -

dist2[0:5,:]

a = np.array([[1,2,3],[4,5,6]])
print(a**2)
print(np.sum(a))
print(np.sum(a, axis = 0))#0번째 index = row에 대해서 더해라 = 세로방향
print(np.sum(a, axis = 1))#1번째 index = col에 대해서 더해라 = 가로방향

a = np.array([[1,2],[3,4]])
print(a)
print(np.matlib.repmat(a,2,3))


