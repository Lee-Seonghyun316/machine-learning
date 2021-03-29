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

a1 = [1,2,3] #실행은 shif+엔터 또는 ctrl+enter 또는 Run
a = np.array([1,2,3]) #1차원 벡터를 생성
a
b = np.array([[1,2],[3,4]]) #2차원 행렬 생성
b
a.shape #shape = dimension (3,) = 1차원 크기가 3인 벡터
b.shape
c = np.array([[1,2,3],[4,5,6]]) #내가 직접 값을 넣어 초기화
c.shape
d = np.zeros((3,4)) #다 0으로 차있는 array size = (3,4)
d

x = np.ones((2,3))#다 1로 차있는
x

y = np.full((2,5),100) #100으로 다 채워라 
y

x = np.arange(0,10,2)#시작, 마지막(포함안됨), 증가량
x

x = np.arange(0,10,1)#시작, 마지막(포함안됨), 증가량
x

x = np.arange(0,10,)#시작, 마지막(포함안됨), 증가량(명시안함 기본값 1)
x

y = np.linspace(0,10,5)#간격을 균등하게 시작하는값, 마지막 값(포함), 개수
y

z = np.linspace(0,10,num = 5, endpoint=False, retstep = False)
#옵션을 더 줄 수 있는 마지막 값 포함 여부 등
z

w = np.array([y, z])
w

np.random.rand(2,3)# 0에서 1사이 값을 랜덤으로 지정

z.ndim #차원의 수, 1=벡터, 2= 행렬 
z.shape #(행, 렬)
z.size # 배열 요소의 총 개수 , 행*열

# +
#나머지 읽어보기
# -

a = np.arange(0,12,1)
a

b = a.reshape(2,6) 
b
b.ndim
#b.shape
#b #순서를 기억해보자
#mat = mat.reshape(2,6)

b[0,0]#1행 1열에 있는 값


b[3,1]

b[0,0:2] #1행의 1열부터 2열까지(마지막 뺴고)

b[1:2,1]

b[1:,1]#비우면 끝까지라는 뜻

b[2,0:]

b[2,-1]

b[3,-2]

b[3,-3:-1]#잘 생각해 보기

b[1:-1,1]

b[1:3,1]

b[[1,2],1]#2,3행 2열

import matplotlib.pyplot as plt
import math

x=np.linspace(0,10,100)
y = 2*x
plt.plot(x,y)
plt.plot([1,2,3,4,5,6],[4,5,6,1,2,3])

#theta = np.linspace(0.2*math.pi,100)
#원그리기 다음에


#히스토그램:도수 분포의 상태를 막대모양으로 표현한 그래프
x = np.random.rand(10000,1)
num_bias = 10
plt.hist(x, num_bias, facecolor='blue')
plt.show()


