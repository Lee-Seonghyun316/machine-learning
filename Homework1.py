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
#Homework1

import numpy as np
y = np.arange(0,8,0.1)
x = y.reshape(8,10)
x
#x = np.arange(0,8,0.1) : 0부터 7.9까지 0.1간격으로 배열을 만들기 위해서 작성
# x.reshape(8,10) : 8행 10열의 행렬로 만들어 주기 위한 reshape
# -

x.ndim
#차원의 수 : 현재 2차원 행렬입니다. 

x.shape
# (행, 열)이 출력되는 명령어로 현재 y는 8행 10열이다. 

x.size
#배열 요소의 총 개수로 행*열의 개수가 출력된다. 
#x는 현재 8*10 = 80개의 요소를 가지고 있다. 

x[2,3]
#3행 4열 출력
#index가 0부터 시작하므로 대괄호 안의 값에서 1씩 [행, 열]의 요소가 출력된다.

x[3:5,4]
#5열의 4행부터 5행까지 출력된다. : 뒤에 오는 숫자바로 전까지만 포함하기 때문이다.  

x[-1,-2]
#뒤에서부터 첫번째 행, 뒤에서부터 두번쨰 열이 출력된다. 

x[2:-3,8:]
#3행부터 뒤에서 네번째 행까지, 9열부터 끝 열까지 출력된다. : 뒤에 오는 숫자바로 전까지만 포함하기 때문에, 여기서는 -3보다 1작은 -4까지 생각한다.   


