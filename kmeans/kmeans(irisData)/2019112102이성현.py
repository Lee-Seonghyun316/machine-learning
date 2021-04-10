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

# 답은 맨 아래에 있습니다..
from sklearn.datasets import load_iris
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import multivariate_normal
from copy import deepcopy
iris = load_iris()
x = iris.data.T
D,N = x.shape

# initialize
K = 3
maxIter = 10
mu0 = np.array([[4.0, 5.0, 6.0],[4.0, 2.0, 3.0],[1.0, 2.0, 3.0],[0.2, 1.0, 2.0]])
muTrail = np.empty((D,K,maxIter))
muTrail[:] = np.nan
muTrail[:,:,0] = mu0
plt.figure(figsize = (15,15))
for i in range(0,D):
    for j in range(0,D):
        plt.subplot(D,D,i*D + j + 1)
        plt.plot(x[i,:],x[j,:],".",markersize = 3,color = [0.75,0.75,0.75])
        for k in range(0,K):
            plt.plot(muTrail[i,k,:],muTrail[j,k,:],'C%dx--'%k)
        plt.xlabel('%d-dim'%(i+1))
        plt.ylabel('%d-dim'%(j+1))
        plt.tight_layout()
plt.savefig('iris.png')


# +
#거리 계산하기

def distance(a, b):
    return sum([(el_a - el_b)**2 for el_a, el_b in list(zip(a, b))]) ** 0.5


# +
#각 데이터 포인트를 그룹화 할 labels생성(0,1,또는 2)
labels = []

for i in range(0,D):
    for j in range(0,D):
            labels.append(np.zeros(N))	# 초기 거리는 모두 0으로 초기화 해줍니다
labels = np.array(labels)
#print(labels)


sepal_length_width=[]
centroids=[]
for i in range(0,D):
    for j in range(0,D):
        sepal_length_width.append(np.array(list(zip(x[i,:],x[j,:]))))

for i in range(0,D):
    for j in range(0,D):
        centroids.append(np.array(list(zip(mu0[i,:],mu0[j,:]))))
centroids = np.array(centroids)

set_centroids=[]
for i in range(0,maxIter):
    set_centroids.append(deepcopy(centroids))
set_centroids= np.array(set_centroids)

sepal_length_width =np.array(sepal_length_width)

# +
distances = []
set_distances = []

for k in range(0, N):
    distances.append(np.zeros(K))
distances = np.array(distances)
print(distances.shape)


for i in range(0,D):
    for j in range(0,D):
        set_distances.append(deepcopy(distances))
set_distances = np.array(set_distances, dtype = object)  
print(set_distances.shape)
# -

##
for k in range(0,16):
    for i in range(N):
        for j in range(K):
            #print(sepal_length_width[k][i],centroids[k][j])
            set_distances[k][i][j]= distance(sepal_length_width[k][i], centroids[k][j])
print(set_distances)

# +
##
cluster = []

for i in range(0,D):
    for j in range(0,D):
            cluster.append(np.zeros(N))	# 초기 거리는 모두 0으로 초기화 해줍니다
cluster = np.array(cluster)
print(cluster.shape)

for i in range(0,D*D):
    for n in range(0,N):
        cluster[i][n] = np.argmin(set_distances[i][n])
        #print(cluster[i][n])

print(cluster)

for i in range(0,D*D):
    for n in range(0,N):
        labels[i][n] = cluster[i][n]

print(labels)

# +
##
set_sum = []
for i in range(0,D*D):
    set_sum.append(np.zeros(K))
set_sum = np.array(set_sum, dtype = object)

# print(set_sum.shape)

count = []
for i in range(0,D*D):
    count.append(np.zeros(K))
# count = np.array(count)

# print(count.shape)

set_mean = []
for i in range(0,D*D):
    set_mean.append(np.zeros(K))
set_mean = np.array(set_mean, dtype = object)
print("mean",set_mean.shape)

print("cluster",cluster)
for i in range(0,D*D):
    for n in range(0,N):
        for k in range(0,K):
           if cluster[i][n]==k:
                #print("cluster:",cluster[i][n])
                set_sum[i][k] += sepal_length_width[i][n]
                count[i][k] += 1

                
print("count",count)

print("set_sum",set_sum)

for i in range(0,D*D):
    for k in range(0,K):
        set_mean[i][k] = set_sum[i][k]/count[i][k]
    
set_mean = np.array(set_mean)    
print("set_mean",set_mean)


# -

##
for d in range(D*D):
    for k in range(K):
        for i in range(0,2):
            set_centroids[1][d][k][i] = set_mean[d][k][i]


# +
for k in range(0,16):
    for i in range(N):
        for j in range(K):
            #print(sepal_length_width[k][i],centroids[k][j])
            set_distances[k][i][j]= distance(sepal_length_width[k][i], set_centroids[1][k][j])

for i in range(0,D*D):
    for n in range(0,N):
        cluster[i][n] = np.argmin(set_distances[i][n])
        #print(cluster[i][n])


for i in range(0,D*D):
    for k in range(0,K):
        set_sum[i][k] = 0
        count[i][k] = 0
        set_mean[i][k] = 0
                
for i in range(0,D*D):
    for n in range(0,N):
        for k in range(0,K):
           if cluster[i][n]==k:
                #print("cluster:",cluster[i][n])
                set_sum[i][k] += sepal_length_width[i][n]
                count[i][k] += 1
                
for i in range(0,D*D):
    for k in range(0,K):
        set_mean[i][k] = set_sum[i][k]/count[i][k]
    
for d in range(D*D):
    for k in range(K):
        for i in range(0,2):
            set_centroids[2][d][k][i] = set_mean[d][k][i]

print(set_centroids)

# +
for k in range(0,16):
    for i in range(N):
        for j in range(K):
            ##
            set_distances[k][i][j]= distance(sepal_length_width[k][i], set_centroids[2][k][j])

for i in range(0,D*D):
    for n in range(0,N):
        cluster[i][n] = np.argmin(set_distances[i][n])
        #print(cluster[i][n])


for i in range(0,D*D):
    for k in range(0,K):
        set_sum[i][k] = 0
        count[i][k] = 0
        set_mean[i][k] = 0
                
for i in range(0,D*D):
    for n in range(0,N):
        for k in range(0,K):
           if cluster[i][n]==k:
                #print("cluster:",cluster[i][n])
                set_sum[i][k] += sepal_length_width[i][n]
                count[i][k] += 1
                
for i in range(0,D*D):
    for k in range(0,K):
        set_mean[i][k] = set_sum[i][k]/count[i][k]
    
for d in range(D*D):
    for k in range(K):
        for i in range(0,2):
            ##
            set_centroids[3][d][k][i] = set_mean[d][k][i]
print(set_centroids)

# +
for k in range(0,16):
    for i in range(N):
        for j in range(K):
            ##
            set_distances[k][i][j]= distance(sepal_length_width[k][i], set_centroids[3][k][j])

for i in range(0,D*D):
    for n in range(0,N):
        cluster[i][n] = np.argmin(set_distances[i][n])
        #print(cluster[i][n])


for i in range(0,D*D):
    for k in range(0,K):
        set_sum[i][k] = 0
        count[i][k] = 0
        set_mean[i][k] = 0
                
for i in range(0,D*D):
    for n in range(0,N):
        for k in range(0,K):
           if cluster[i][n]==k:
                #print("cluster:",cluster[i][n])
                set_sum[i][k] += sepal_length_width[i][n]
                count[i][k] += 1
                
for i in range(0,D*D):
    for k in range(0,K):
        set_mean[i][k] = set_sum[i][k]/count[i][k]
    
for d in range(D*D):
    for k in range(K):
        for i in range(0,2):
            ##
            set_centroids[4][d][k][i] = set_mean[d][k][i]


# +
for k in range(0,16):
    for i in range(N):
        for j in range(K):
            ##
            set_distances[k][i][j]= distance(sepal_length_width[k][i], set_centroids[4][k][j])

for i in range(0,D*D):
    for n in range(0,N):
        cluster[i][n] = np.argmin(set_distances[i][n])
        #print(cluster[i][n])


for i in range(0,D*D):
    for k in range(0,K):
        set_sum[i][k] = 0
        count[i][k] = 0
        set_mean[i][k] = 0
                
for i in range(0,D*D):
    for n in range(0,N):
        for k in range(0,K):
           if cluster[i][n]==k:
                #print("cluster:",cluster[i][n])
                set_sum[i][k] += sepal_length_width[i][n]
                count[i][k] += 1
                
for i in range(0,D*D):
    for k in range(0,K):
        set_mean[i][k] = set_sum[i][k]/count[i][k]
    
for d in range(D*D):
    for k in range(K):
        for i in range(0,2):
            ##
            set_centroids[5][d][k][i] = set_mean[d][k][i]


# +
for k in range(0,16):
    for i in range(N):
        for j in range(K):
            ##
            set_distances[k][i][j]= distance(sepal_length_width[k][i], set_centroids[5][k][j])

for i in range(0,D*D):
    for n in range(0,N):
        cluster[i][n] = np.argmin(set_distances[i][n])
        #print(cluster[i][n])


for i in range(0,D*D):
    for k in range(0,K):
        set_sum[i][k] = 0
        count[i][k] = 0
        set_mean[i][k] = 0
                
for i in range(0,D*D):
    for n in range(0,N):
        for k in range(0,K):
           if cluster[i][n]==k:
                #print("cluster:",cluster[i][n])
                set_sum[i][k] += sepal_length_width[i][n]
                count[i][k] += 1
                
for i in range(0,D*D):
    for k in range(0,K):
        set_mean[i][k] = set_sum[i][k]/count[i][k]
    
for d in range(D*D):
    for k in range(K):
        for i in range(0,2):
            ##
            set_centroids[6][d][k][i] = set_mean[d][k][i]


# +
for k in range(0,16):
    for i in range(N):
        for j in range(K):
            ##
            set_distances[k][i][j]= distance(sepal_length_width[k][i], set_centroids[6][k][j])

for i in range(0,D*D):
    for n in range(0,N):
        cluster[i][n] = np.argmin(set_distances[i][n])
        #print(cluster[i][n])


for i in range(0,D*D):
    for k in range(0,K):
        set_sum[i][k] = 0
        count[i][k] = 0
        set_mean[i][k] = 0
                
for i in range(0,D*D):
    for n in range(0,N):
        for k in range(0,K):
           if cluster[i][n]==k:
                #print("cluster:",cluster[i][n])
                set_sum[i][k] += sepal_length_width[i][n]
                count[i][k] += 1
                
for i in range(0,D*D):
    for k in range(0,K):
        set_mean[i][k] = set_sum[i][k]/count[i][k]
    
for d in range(D*D):
    for k in range(K):
        for i in range(0,2):
            ##
            set_centroids[7][d][k][i] = set_mean[d][k][i]


# +
for k in range(0,16):
    for i in range(N):
        for j in range(K):
            ##
            set_distances[k][i][j]= distance(sepal_length_width[k][i], set_centroids[7][k][j])

for i in range(0,D*D):
    for n in range(0,N):
        cluster[i][n] = np.argmin(set_distances[i][n])
        #print(cluster[i][n])


for i in range(0,D*D):
    for k in range(0,K):
        set_sum[i][k] = 0
        count[i][k] = 0
        set_mean[i][k] = 0
                
for i in range(0,D*D):
    for n in range(0,N):
        for k in range(0,K):
           if cluster[i][n]==k:
                #print("cluster:",cluster[i][n])
                set_sum[i][k] += sepal_length_width[i][n]
                count[i][k] += 1
                
for i in range(0,D*D):
    for k in range(0,K):
        set_mean[i][k] = set_sum[i][k]/count[i][k]
    
for d in range(D*D):
    for k in range(K):
        for i in range(0,2):
            ##
            set_centroids[8][d][k][i] = set_mean[d][k][i]


# +
for k in range(0,16):
    for i in range(N):
        for j in range(K):
            ##
            set_distances[k][i][j]= distance(sepal_length_width[k][i], set_centroids[8][k][j])

for i in range(0,D*D):
    for n in range(0,N):
        cluster[i][n] = np.argmin(set_distances[i][n])
        #print(cluster[i][n])


for i in range(0,D*D):
    for k in range(0,K):
        set_sum[i][k] = 0
        count[i][k] = 0
        set_mean[i][k] = 0
                
for i in range(0,D*D):
    for n in range(0,N):
        for k in range(0,K):
           if cluster[i][n]==k:
                #print("cluster:",cluster[i][n])
                set_sum[i][k] += sepal_length_width[i][n]
                count[i][k] += 1
                
for i in range(0,D*D):
    for k in range(0,K):
        set_mean[i][k] = set_sum[i][k]/count[i][k]
    
for d in range(D*D):
    for k in range(K):
        for i in range(0,2):
            ##
            set_centroids[9][d][k][i] = set_mean[d][k][i]
print(set_centroids.shape)
# -

plt.figure(figsize = (15,15))
for i in range(0,D):
    for j in range(0,D):
        plt.subplot(D,D,i*D + j + 1)
        plt.plot(x[i,:],x[j,:],".",markersize = 3,color = [0.75,0.75,0.75])
        for k in range(0,K):
            for n in range(0,10):
                plt.plot(set_centroids[n][D*i+j][k][0],set_centroids[n][D*i+j][k][1],'C%dx--'%k)
        plt.xlabel('%d-dim'%(i+1))
        plt.ylabel('%d-dim'%(j+1))
        plt.tight_layout()
plt.savefig('iris.png')
