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
import matplotlib
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data.T
t = iris.target
D,N = x.shape

# 총 4차원의 input dimension 중에서
# 1, 2차원만 선택해서 input으로 쓰겠음
x1 = x[0,:]
x2 = x[1,:]

# 이번에 보여주는 수업 예제랑
# HW04랑 notation 다릅니다!!!!!!! 
# 주의하세요!!!

# multi-class (3-class) 문제
print(t)

# two-class, 0이냐 vs. 아니냐
t[np.where(t != 0)] = -1
t[np.where(t == 0)] = +1
# perceptron에 적용하기 위해서

print(t)

# +
# w를 초기화
w = np.array([-10, 1.0, 1.5]) #3차원 = input 2차원 + bias 1차원

cmap_sample = matplotlib.colors.ListedColormap(['black', 'red'])
cmap_region = matplotlib.colors.ListedColormap(['gray', 'white'])

# 전체 면적에 뿌려놓은 모든 점들
x1_new = (np.linspace(np.min(x1),np.max(x1),100))
x2_new = (np.linspace(np.min(x2),np.max(x2),100))

[xx1,xx2] = np.meshgrid(x1_new, x2_new)

# y_new = 전체 면적에 뿌려놓은 점들에 대해 평가한 값
y_new = w[0] + w[1]*xx1 + w[2]*xx2
# y_predict = 전체 샘플(점들)에 대해 평가한 값
y_predict = w[0] + w[1]*x1 + w[2]*x2

# perceptron method
# misclassified pattern
id_misclass = np.where(y_predict * t < 0)[0]
print(id_misclass)


plt.figure(figsize = (10,5))
plt.pcolormesh(xx1,xx2,y_new > 0, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
plt.scatter(x1,x2, c = t, s = 20, cmap = cmap_sample, vmin = -1, vmax = +1)
plt.scatter(x1[id_misclass], x2[id_misclass], s = 50, marker = 'o', facecolor = 'none', edgecolor = 'cyan')

# +
eta = 0.002
maxIter = 10000

cost = np.zeros(maxIter)
cost[:] = np.nan

plt.figure(figsize = (10,5))
plt.pcolormesh(xx1,xx2,y_new > 0, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
plt.scatter(x1,x2, c = t, s = 20, cmap = cmap_sample, vmin = -1, vmax = +1)
plt.scatter(x1[id_misclass], x2[id_misclass], s = 50, marker = 'o', facecolor = 'none', edgecolor = 'cyan')

# stochastic gradient descent
for iter in range(0,maxIter):
    # 한 싸이클, 한 iteration
    id_misclass = np.random.permutation(id_misclass)
#     print(id_misclass[0])
    
    # misclassified, eta*(x_n * t_n)
    # 우리가 사용할 sample 한개 = id_misclass[0]
    w = w + eta*np.array([1, x1[id_misclass[0]], x2[id_misclass[0]]])*t[id_misclass[0]]
    
    # 전체 영역에 뿌린 점들에 대해 update
    y_new = w[0] + w[1]*xx1 + w[2]*xx2
    # 전체 샘플에 대해 update
    y_predict = (w[0] + w[1]*x1 + w[2]*x2)    
    
    id_misclass = np.where(y_predict*t < 0)[0]   
    cost[iter] = -np.sum(y_predict[id_misclass]*t[id_misclass])     

plt.figure(figsize = (10,5))
plt.pcolormesh(xx1,xx2,y_new > 0, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
plt.scatter(x1,x2, c = t, s = 20, cmap = cmap_sample, vmin = -1, vmax = +1)
plt.scatter(x1[id_misclass], x2[id_misclass], s = 50, marker = 'o', facecolor = 'none', edgecolor = 'cyan')

plt.figure(figsize = (5,5))
plt.plot(cost)


# +
# logistic regression
def sigmoid(x):
    return 1 / (1 +np.exp(-x))

t[np.where(t < 0)] = 0 # sample: black, region: grey
t[np.where(t > 0)] = +1 # sample: red, region: white

w = np.array([-10, 1.0, 1.5])

# w^T x, sigmoid 함수 안에 입력할 값 a
# sigmoid 함수에 a를 넣어서 나온 값 y
a_new = w[0] + w[1]*xx1 + w[2]*xx2
y_new = sigmoid(a_new)

# _new = 전체 영역에 뿌린 점들에 대한 = 그림그리기용
# _predict = 전체 샘플에 대한
a_predict = (w[0] + w[1]*x1 + w[2]*x2)
y_predict = sigmoid(a_predict)

plt.figure(figsize = (10,5))
plt.pcolormesh(xx1,xx2,y_new > 0.5, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
plt.scatter(x1,x2, c = t, s = 20, cmap = cmap_sample, vmin = 0, vmax = +1)

# +
eta = 0.0001
maxIter = 10000

cost = np.zeros(maxIter)
cost[:] = np.nan

plt.figure(figsize = (10,5))
plt.pcolormesh(xx1,xx2,y_new > 0.5, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
plt.scatter(x1,x2, c = t, s = 20, cmap = cmap_sample, vmin = 0, vmax = +1)

for iter in range(0,maxIter):
    # gradient = \sum_{n=1}^{N} (y_n - t_n)*x_n
    # (y_predict - t) = 1 x N 벡터
    # x_n = 3 x N 벡터
    # np.sum(, axis = 1) = 3 x 1 벡터
    w = w - eta*np.sum((y_predict - t) * np.vstack([np.ones((1,N)), x1.reshape(1,-1), x2.reshape(1,-1)]), axis = 1)
        
    a_new = w[0] + w[1]*xx1 + w[2]*xx2
    y_new = sigmoid(a_new)

    a_predict = (w[0] + w[1]*x1 + w[2]*x2)
    y_predict = sigmoid(a_predict)
    
    # cross-entropy
    cost[iter] = -np.sum(t*np.log(y_predict) + (1 - t)*np.log(1 - y_predict))   

plt.figure(figsize = (10,5))
plt.pcolormesh(xx1,xx2,y_new > 0.5, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
plt.scatter(x1,x2, c = t, s = 20, cmap = cmap_sample, vmin = 0, vmax = +1)

plt.figure(figsize = (5,5))
plt.plot(cost)
