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
import matplotlib.pyplot as plt

x1 = np.random.rand(100, 1)
y = 2.0 + 3*x1 + 0.2*np.random.randn(100,1)

plt.figure(figsize = (5,3))
plt.scatter(x1,y, s=10)
plt.xlabel('x_1')
plt.ylabel('y')
plt.tight_layout()

# design matrix
m,d = x1.shape # m개의 sample, d-차원
n = d + 1 # d + bias 1차원 = n-차원

X = np.hstack((np.ones((m,1)), x1))
y = y

theta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
y_predict = X.dot(theta_hat)
error = y - y_predict

print(theta_hat)

x_new = np.linspace(0,1,100)
X_new = np.hstack((np.ones((100,1)), x_new.reshape(-1,1)))
y_new = X_new.dot(theta_hat)

plt.plot(x_new,y_new,'x-')

# -


