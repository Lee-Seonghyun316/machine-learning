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
import matplotlib
import matplotlib.pyplot as plt

t1 = np.array([-1, 2])
t2 = np.array([2, -1.5])
t3 = np.array([20, -15])

r1 = np.array([[1,2],[2,2]])
r2 = np.array([[1,2],[2,1]])
r3 = np.array([[1,0],[0,1]])

n1 = 100
n2 = 80
n3 = 0

x1 = np.random.randn(2,n1)
x1 = r1.dot(x1) + t1.reshape(2,-1)

x2 = np.random.randn(2,n2)
x2 = r2.dot(x2) + t2.reshape(2,-1)

x3 = np.random.randn(2,n3)
x3 = r3.dot(x3) + t3.reshape(2,-1)

plt.figure(figsize = (5,5))
plt.scatter(x1[0,:],x1[1,:], marker = "o", s = 10, color = 'k')
plt.scatter(x2[0,:],x2[1,:], marker = "o", s = 10, color = 'r')
plt.scatter(x3[0,:],x3[1,:], marker = "o", s = 10, color = 'r')
plt.savefig('data.png')

# +
# least square
# preprocess
x = np.ones((1,n1+n2+n3))
x = np.vstack([x,np.hstack([x1,x2,x3])])

t = np.hstack([np.ones((1,n1)), np.zeros((1,n2+n3))])

y = np.hstack([np.ones((1,n1)), np.zeros((1,n2+n3))])
y = np.vstack([y,np.hstack([np.zeros((1,n1)), np.ones((1,n2+n3))])])

# initialize
xx1 = (np.linspace(np.min(x[1,:]),np.max(x[1,:]),100))
xx2 = (np.linspace(np.min(x[2,:]),np.max(x[2,:]),100))

[xx1,xx2] = np.meshgrid(xx1, xx2)

cmap_sample = matplotlib.colors.ListedColormap(['red', 'black'])
cmap_region = matplotlib.colors.ListedColormap(['white', 'gray'])

# implement

# visualize
plt.figure(figsize = (5,5))
plt.pcolormesh(xx1,xx2,y1_new - y2_new > 0, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
plt.scatter(x[:,1],x[:,2], c = t, s = 10, cmap = cmap_sample, vmin = 0, vmax = +1)
plt.savefig('least.png')

# +
# perceptron
# preprocess
x = np.hstack([x1,x2,x3])
y = np.hstack([np.ones((1,n1)), -np.ones((1,n2+n3))])

xx1 = (np.linspace(np.min(x[0,:]),np.max(x[0,:]),100))
xx2 = (np.linspace(np.min(x[1,:]),np.max(x[1,:]),100))

[xx1,xx2] = np.meshgrid(xx1, xx2)

cmap_sample = matplotlib.colors.ListedColormap(['red', 'black'])
cmap_region = matplotlib.colors.ListedColormap(['white', 'gray'])

# initialize
w = np.array([-10, 1.0, 1.5])

y_new = w[0] + w[1]*xx1 + w[2]*xx2
y_predict = (w[0] + w[1]*x[0,:] + w[2]*x[1,:])

id_misclass = np.where(y_predict*y < 0)
id_misclass = np.random.permutation(id_misclass[0])

eta = 0.01
maxIter = 1000

cost = np.zeros(maxIter)
cost[:] = np.nan

# implement
for iter in range(0,maxIter):
    
    if id_misclass.size == 0:
        break
    
# visualize
plt.figure(figsize = (5,5))
plt.pcolormesh(xx1,xx2,y_new > 0, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
plt.scatter(x[0,:],x[1,:], c = y, s = 10, cmap = cmap_sample, vmin = -1, vmax = +1)
plt.scatter(x[0,id_misclass],x[1,id_misclass], s = 20, marker = 'o', facecolor = 'none', edgecolor = 'cyan')
plt.savefig('perceptron.png')

plt.figure(figsize = (5,5))
plt.plot(cost)
plt.savefig('perceptron-cost.png')


# +
# logistic regression
def sigmoid(x):
    return 1 / (1 +np.exp(-x))

# preprocess
x = np.hstack([x1,x2,x3])
t = np.hstack([np.ones((1,n1)), np.zeros((1,n2+n3))])

xx1 = (np.linspace(np.min(x[0,:]),np.max(x[0,:]),100))
xx2 = (np.linspace(np.min(x[1,:]),np.max(x[1,:]),100))

[xx1,xx2] = np.meshgrid(xx1, xx2)

cmap_sample = matplotlib.colors.ListedColormap(['red', 'black'])
cmap_region = matplotlib.colors.ListedColormap(['white', 'gray'])

N = n1 + n2 + n3

# initialize
w = np.array([-10, 1.0, 1.5])
a_new = w[0] + w[1]*xx1 + w[2]*xx2
y_new = sigmoid(a_new)

a_predict = (w[0] + w[1]*x[0,:] + w[2]*x[1,:])
y_predict = sigmoid(a_predict)

eta = 0.01
maxIter = 100

cost = np.zeros(maxIter)
cost[:] = np.nan

# implement
for iter in range(0,maxIter):

# visualize
plt.figure(figsize = (5,5))
plt.pcolormesh(xx1,xx2,y_new > 0.5, cmap=cmap_region, vmin = 0.0, vmax = 1.0)
plt.scatter(x[0,:],x[1,:], c = t, s = 10, cmap = cmap_sample, vmin = 0, vmax = +1)
plt.savefig('logistic.png')

plt.figure(figsize = (5,5))
plt.plot(cost)
plt.savefig('logistic-cost.png')
