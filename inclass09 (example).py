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
y = iris.target
D,N = x.shape

print(x.shape)
print(y.shape)

x1 = x[0,:]
x2 = x[1,:]

y[np.where(y != 0)] = -1 # sample: black, region: grey
y[np.where(y == 0)] = +1 # sample: red, region: white

x1_new = (np.linspace(np.min(x1),np.max(x1),100))
x2_new = (np.linspace(np.min(x2),np.max(x2),100))

[uu,vv] = np.meshgrid(x1_new, x2_new)

cmap_sample = matplotlib.colors.ListedColormap(['black', 'red'])
cmap_region = matplotlib.colors.ListedColormap(['gray', 'white'])

# w = np.random.randn(3)
w = np.array([-10, 1.0, 1.5])

y_new = w[0] + w[1]*uu + w[2]*vv
y_predict = (w[0] + w[1]*x1 + w[2]*x2)

id_misclass = np.where(y_predict*y < 0)
id_misclass = np.random.permutation(id_misclass[0])

plt.figure(figsize = (10,5))
plt.pcolormesh(uu,vv,y_new > 0, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
plt.scatter(x1,x2, c = y, s = 20, cmap = cmap_sample, vmin = -1, vmax = +1)
plt.scatter(x1[id_misclass],x2[id_misclass], s = 50, marker = 'o', facecolor = 'none', edgecolor = 'cyan')

# +
eta = 0.01
maxIter = 100

cost = np.zeros(maxIter)
cost[:] = np.nan

for iter in range(0,maxIter):
    w = w + eta*np.array([1, x1[id_misclass[0]], x2[id_misclass[0]]])*y[id_misclass[0]]
    
    y_new = w[0] + w[1]*uu + w[2]*vv
    y_predict = (w[0] + w[1]*x1 + w[2]*x2)
    
    id_misclass = np.where(y_predict*y < 0)[0]    
    
    cost[iter] = -np.sum(y_predict[id_misclass]*y[id_misclass])    
    id_misclass = np.random.permutation(id_misclass)
    
    plt.figure(figsize = (10,5))
    plt.pcolormesh(uu,vv,y_new > 0, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
    plt.scatter(x1,x2, c = y, s = 20, cmap = cmap_sample, vmin = -1, vmax = +1)
    plt.scatter(x1[id_misclass],x2[id_misclass], s = 50, marker = 'o', facecolor = 'none', edgecolor = 'cyan')
    
    if id_misclass.size == 0:
        break
    
plt.figure(figsize = (10,5))
plt.pcolormesh(uu,vv,y_new > 0, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
plt.scatter(x1,x2, c = y, s = 20, cmap = cmap_sample, vmin = -1, vmax = +1)
plt.scatter(x1[id_misclass],x2[id_misclass], s = 50, marker = 'o', facecolor = 'none', edgecolor = 'cyan')

plt.figure()
plt.plot(cost)


# +
def sigmoid(x):
    return 1 / (1 +np.exp(-x))

t = y
t[np.where(t < 0)] = 0 # sample: black, region: grey
t[np.where(t > 0)] = +1 # sample: red, region: white

cmap_sample = matplotlib.colors.ListedColormap(['black', 'red'])
cmap_region = matplotlib.colors.ListedColormap(['gray', 'white'])

w = np.array([-10, 1.0, 1.5])
a_new = w[0] + w[1]*uu + w[2]*vv
y_new = sigmoid(a_new)

a_predict = (w[0] + w[1]*x1 + w[2]*x2)
y_predict = sigmoid(a_predict)

plt.figure(figsize = (10,5))
plt.pcolormesh(uu,vv,y_new > 0.5, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
plt.scatter(x1,x2, c = t, s = 20, cmap = cmap_sample, vmin = 0, vmax = +1)

# +
eta = 0.001
maxIter = 1000

cost = np.zeros(maxIter)
cost[:] = np.nan

for iter in range(0,maxIter):
    cost[iter] = -np.sum(t*np.log(y_predict) + (1 - t)*np.log(1 - y_predict))
    w = w - eta*np.sum((y_predict - t)*np.vstack([np.ones((1,N)), x1.reshape(1,-1), x2.reshape(1,-1)]), axis = 1)

    a_new = w[0] + w[1]*uu + w[2]*vv
    y_new = sigmoid(a_new)

    a_predict = (w[0] + w[1]*x1 + w[2]*x2)
    y_predict = sigmoid(a_predict)

#     plt.figure(figsize = (10,5))
#     plt.pcolormesh(uu,vv,y_new > 0.5, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
#     plt.scatter(x1,x2, c = t, s = 20, cmap = cmap_sample, vmin = 0, vmax = +1)    
    
plt.figure(figsize = (10,5))
plt.pcolormesh(uu,vv,y_new > 0.5, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
plt.scatter(x1,x2, c = t, s = 20, cmap = cmap_sample, vmin = 0, vmax = +1)
    
plt.figure()
plt.plot(cost)
