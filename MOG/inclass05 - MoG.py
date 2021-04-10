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
from scipy.stats import multivariate_normal
from matplotlib import colors

x1 = np.array([[2.5, -1],[-1, 1.3]]).dot(np.random.randn(2, 1000)) + np.matlib.repmat(np.array([6, 7]).reshape(2,1),1,1000)
x2 = np.array([[1.3, 0.5],[0.5, -2.9]]).dot(np.random.randn(2, 500)) + np.matlib.repmat(np.array([0, -2]).reshape(2,1),1,500)
x3 = np.array([[1.5, 0.2],[0.2, 1.76]]).dot(np.random.randn(2,500)) + np.matlib.repmat(np.array([-5, 10]).reshape(2,1),1,500)

cmap = colors.ListedColormap(['C0','C1','C2'])

plt.figure(figsize = (6,3))
plt.subplot(1,2,1)
plt.plot(x1[0,:],x1[1,:],"C0.",markersize = 1)
plt.plot(x2[0,:],x2[1,:],"C1.",markersize = 1)
plt.plot(x3[0,:],x3[1,:],"C2.",markersize = 1)

x = np.hstack((x1, x2, x3))
 
pi = np.array([1,1,1])
pi = pi / sum(pi)
mu = 20*(np.random.rand(2,3) - 1/2)
sigma = np.zeros((2,2,3))
for cluster in range(0,3):
    sigma[:,:,cluster] = np.array([[1,0],[0,1]])
    
u = np.linspace(-15,15,100)
v = np.linspace(-15,15,100)
uu, vv = np.meshgrid(u, v)
mog_pdf = np.zeros(uu.shape)
for cluster in range(0,3):
    pdf = multivariate_normal.pdf(np.hstack((uu.reshape(-1,1),vv.reshape(-1,1))), mean = mu[:,cluster], cov = sigma[:,:,cluster])
    mog_pdf += pi[cluster] * pdf.reshape(100,100)

plt.subplot(1,2,2)
plt.plot(x[0,:],x[1,:],".",markersize = 1, color = [0.75, 0.75, 0.75])
plt.contour(uu,vv,mog_pdf)
plt.savefig('mog-iter0.png',format='png')

# +
row,col = x.shape

for iter in range(1,20):
    gamma = np.zeros((3,col))
    for cluster in range(0,3):
        gamma[cluster,:] = pi[cluster] * multivariate_normal.pdf(x.reshape(2,-1).T, mean = mu[:,cluster], cov = sigma[:,:,cluster])

    gamma = gamma / np.matlib.repmat(np.sum(gamma,axis=0),3,1)
    idmax = np.argmax(gamma,axis=0)


    plt.figure(figsize =(6,3))
    plt.subplot(1,2,1)
    plt.scatter(x[0,:],x[1,:],c = idmax,s = 0.1, cmap = cmap)
    plt.plot(mu[0,0],mu[1,0],"C0x",markersize = 10)
    plt.plot(mu[0,1],mu[1,1],"C1x",markersize = 10)
    plt.plot(mu[0,2],mu[1,2],"C2x",markersize = 10)

    for cluster in range(0,3):
        mu[:,cluster] = np.mean(x[:,idmax == cluster],axis=1)
        res = x[:,idmax == cluster] - np.matlib.repmat(mu[:,cluster].reshape(2,1),1,sum(idmax == cluster))
        sigma[:,:,cluster] = res.dot(res.T) / sum(idmax == cluster)
        pi[cluster] = sum(idmax == cluster) / col


    uu, vv = np.meshgrid(u, v)
    mog_pdf = np.zeros(uu.shape)
    for cluster in range(0,3):
        pdf = multivariate_normal.pdf(np.hstack((uu.reshape(-1,1),vv.reshape(-1,1))), mean = mu[:,cluster], cov = sigma[:,:,cluster])
        mog_pdf += pi[cluster] * pdf.reshape(100,100)

    plt.subplot(1,2,2)
    plt.plot(x[0,:],x[1,:],".",markersize = 1, color = [0.75, 0.75, 0.75])
    plt.contour(uu,vv,mog_pdf)
    plt.plot(mu[0,0],mu[1,0],"C0x",markersize = 10)
    plt.plot(mu[0,1],mu[1,1],"C1x",markersize = 10)
    plt.plot(mu[0,2],mu[1,2],"C2x",markersize = 10)
    
    plt.savefig('mog-iter%d.png' % iter, format='png')
# -


