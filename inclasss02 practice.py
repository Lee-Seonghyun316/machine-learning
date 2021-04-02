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

import numpy as np

# +
a = np.array([1, 2, 3]) #1차원 벡터를 생성
b = np.array([[1, 2],[3, 4]]) #2차원 행렬을 생성
c = np.array([[1,2,3],[4,5,6]])

d = np.zeros((3,4)) #다 0으로 차있는 array, size = (3,4)
x = np.ones((2,3)) #다 1로 차있는
y = np.full((2,5), 200)
# -

y

# x = np.arange(시작하는값, 마지막값(포함x), 증가량(기본값 = 1))
x = np.arange(0, 10)
# y = np.linspace(시작하는값, 마지막값(포함o), 개수)
y = np.linspace(0, 10, 3)
z1 = np.linspace(0, 10, num=5, endpoint=False)
z2 = np.linspace(0, 10, num=5, endpoint=True)

z = np.array([z1, z2])

z

np.random.rand(2,3)

z.ndim #차원의수, 1 = 벡터, 2 = 행렬
z.shape #크기, size (행,) (행, 열)
z.size #배열 요소의 총 개수, 행x열

a = np.arange(0,12,1)
a
b = a.reshape(4,3) # b.ndim = 2, b.shape = (3,4)

b #순서를 잘 기억하세요, 0->11까지인 벡터, 0~3 / 4~7 / 

b[1:-1, 1] 
#b[1:3, 1] = b[[1,2], 1] #2-3행, 2열


# +
import matplotlib.pyplot as plt
import math
x = np.linspace(0,10,100)
y = 2*x

plt.plot(x,y)
plt.plot([1,2,3,4,5,6],[4,5,6,1,2,3])
# -

#히스토그램: 도수 분포의 상태를 막대모양으로 표현한 그래프
x = np.random.rand(1000,2)
#[0,1)사이의 난수 생성, 1000x2(1000행 2열 행렬생성)
num_bins = 100 #몇 개로 쪼갤 것인가     
plt.hist(x, num_bins, facecolor='blue')  
plt.show()
