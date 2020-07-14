# 손글씨 숫자를 학습하는 신경망을 구현해보자. 2층 신경망(은닉층 1개)을 대상으로 MNIST 데이터셋을 사용하여 학습한다.

import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01): # input_size: 입력층 뉴런수, hidden_size: 은닉층 뉴런수, output_size: 출력층 뉴런수
        # 가중치 초기화
        self.params = {} # 신경망의 매개변수(가중치, 편향)을 보관하는 딕셔너리 변수
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predict(self, x): # 인수 x는 이미지 데이터
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grad = {} # 기울기를 보관하는 딕셔너리 변수
        grad['W1'] = numerical_gradient(loss_W, self.params['W1']) # 1번째 층의 가중치의 기울기
        grad['b1'] = numerical_gradient(loss_W, self.params['b1']) # 1번째 층의 편향의 기울기
        grad['W2'] = numerical_gradient(loss_W, self.params['W2']) # 2번쨰 층의 가중치의 기울기
        grad['b2'] = numerical_gradient(loss_W, self.params['b2']) # 2번째 층의 편향의 기울기
        
        return grad