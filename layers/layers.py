import numpy as np
from abc import ABC, abstractmethod

from utils.utils import im2col, col2im

class Layer(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def forward(self, input_data):
        pass


class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((output_size))

    def forward(self, input_data):
        self.input = input_data
        # 1. 출력 계산 : output = input * W + b
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward(self, dout):
        # 1. 가중치 (W)의 기울기 계산: dL/dW = input^T * dout
        dw = np.dot(self.input.T, dout)

        # 2. 편향(b)의 기울기 계산: dL/db = sum(dout)
        db = np.sum(dout, axis=0)

        # 3. 입력(X)의 기울기 계산: dL/dX = dout * W^T
        dx = np.dot(dout, self.weights.T)
        self.grads= [dw, db]

        return dx


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, self.input)

        return self.output
    
    def backward(self, dout):
        dx = dout.copy()
        dx[self.input <= 0] = 0
        return dx
    
class Conv2d(Layer):
    def __iniit__(self, W, b, stride=1, pad=0):
        super().__init__()
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 중간데이터 저장용 (backward 시 사용)  
        self.x = None
        self.col = None
        self.col_W = None

        self.dw = None
        self.db = None

    def forward(self, x):
        # 1. 차원 준비
        self.x = x
        FN, C, FH, FW = self.W.shape # 필터 수, 채널 수, 필터 높이, 필터 너비
        N, C, H, W = x.shape # 배치 크기, 채널 수, 입력 높이, 입력 너비

        # 2. 이미지 펼치기(im2col)
        # 4차원 데이터를 2차원 행렬로 변신
        col = im2col(x, FH, FW, self.stride, self.pad)

        # 3. 필터 펼치기
        col_W = self.W.reshape(FN, -1).T  # (C*FH*FW, FN)

        out = np.dot(col, col_W) + self.b  # (N*out_h*out_w, FN)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out
    
    def backward(self, dout):
        # 1. 차원 준비
        FN, C, FH, FW = self.W.shape
        
        # dout이 4차원으로 오니까, 다시 행렬 연산하게 2차원으로 펼침
        # (N, FN, OH, OW) -> (N*OH*OW, FN)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # 2. 편향 미분 (Linear랑 똑같음: sum)
        self.db = np.sum(dout, axis=0)
        
        # 3. 가중치(필터) 미분 (Linear랑 똑같음: dot)
        self.dW = np.dot(self.col.T, dout)
        
        # 펼쳐진 dW를 원래 필터 모양(4차원)으로 복구
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 4. 입력 데이터 미분 (Linear랑 똑같음: dot)
        dcol = np.dot(dout, self.col_W.T)
        
        # 5. 펼쳐진 데이터를 다시 이미지 모양으로 복구 (col2im)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
