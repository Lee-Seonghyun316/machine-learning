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

#4.5
import numpy as np 
a = np.array([[1,2],[3,4]])
b=np.copy(a)
print(a)
print(b)

a[0,0] = 100
print(a)
print(b)
#깊은 복사 필..

# +
#오른쪽 화살표 눌러서 항상 확인하고 제출(변수 순서 뒤바뀔 때 문제 발생 :사용 후 선언이라던지..)
# 과제 정답 보고 공부하기..
#어려운 코드 문제는 모범 답안을 올려주심..

# +
#4월5일
#inClass9
#iris data로 x, y 생성
#4차원의 input dimensionwnddptj 
#1, 2차원만 선택해서 input으로 쓰겠음
#이번에 보여주ㅡㄴ 수업 예제랑 HW4랑 notation다릅니다. 
#주의 하세요

#y: multi-class(3) -> 2class로 만들기
#y: -1 or 1

#classification의 label을 1 or -1로 바꿈 reason  ->perception

#w 초기화
#그림그리기
#왼쪽 코드가 같이 올라옴.., 강의자료랑 같이

#-1은 red +1 black 으로 그림(모르면 다시 듣기)
#컨투어 사용해서 그리기..

#매시 그리드 사용
#x, y 축 100단계로 조갠 후 매시그리드 생성
#프린트 맨 앞 값이 가로축 최소 맨 위 가로축 최대 (세로 축도 동일) x1
#세로 다르고 가로 같거나 가로같고 세로 다르다. 
#삼차원인 이유는 2차원 + bias 1차원
#전체 면적에 뿌려놓은 모든 점들 x1_new, x2_new에 대해 y_new = w[0]+w[1]*xx1+w[2]xx2

#전체 면적에서 샘플들은 점으로 찍히지만
#메시 그리드로 면으로..? 0보다 크나 작나로 바운드리를 그림
#pcolormesh로 그림 점 위에 같이 그릴 수 있게 된다. 
#검은 애들은 회색에 빨간 애들은 흰색에 다 있어야 하는데.. 그렇게 안되네?
#w가 굉장히 좋지 못한 값이라는 뜻 (맨 윗줄)

#perceptron method에서 
#misiclassified pattern
#y > 0 , t = +1-> y*t>0
#y < 0 , t = -1-> y*t>0
#그래서 여기서도 y-> t로 변경 후 

#y_prediect구한 후 (전체 샘플(점들)에 대해 평가한 값)
#y_new = 전체 면적에 뿌려놓은 점들에 대해 평가한 값

#print(y_predict*t<0) true 이면 잘못 구해짐 id_misclass-> 그리기 : 사이즈가 크고 색은 청록색으로 그림

#라이브러리 긁어 오면 되는데 왜 이짓을 할까? 배운 이론을 코드로 직접 짜보기

#쉬는 시간 후 다시 시작
#실전

#learning 스텝
#cost 정의
#nan값 넣기
#한 사이클 잘 짜고 
#포문 붙여서 잘 짯다는 가정 하에 반복문 돌리기 
#포문 안에 한번돌리는 걸로 일단 짜기 시작(팁)
#stochastic gradient descent
#id_misclass = n.random.permutation(id_misclass)
#섞이고 벡터의 순서가 바뀌니 첫번재 거 뽑으면된다. 
#w * w+eta*
#misclassifiend. eta*(x_n*t_n)
#w w+eta*np.array([1,x1[id_misclass[0]],x2[id_misclass[0]]])
#새로 구한 w 에 대해 전체 영역에 뿌린 점들에 대해 업데이트, 전체 새ㅡㄹ에 대해 업데이트 
#misclass도 다시 구하고 
#cost구하기
#for 문 바깥에 코스트 그려봄-> 점점 내려감, 맨 처음과 맨 끝만 그리기 

#perceptron알고리즘 (gradient descent적용) 직접 그리기 완료

# +
#logistic regression 짜보기
#sigmoid 함수 정의
#여기서부터 다시 듣기 
