import numpy as np


class Adam:
    """
    Adam Optimizer
    
    Args:
        lr: 학습률 (default: 0.001)
        beta1: 1차 모멘트 감쇠 계수 (default: 0.9)
        beta2: 2차 모멘트 감쇠 계수 (default: 0.999)
        epsilon: 수치 안정성을 위한 작은 값 (default: 1e-8)
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1  # momentum 계수
        self.beta2 = beta2  # rmsprop 계수
        self.epsilon = epsilon
        
        # 모멘트 저장 (각 파라미터마다)
        self.m = {}  # 1차 모멘트 (평균)
        self.v = {}  # 2차 모멘트 (분산)
        self.t = 0   # 시간 스텝 (bias correction용)
    
    def update(self, params, grads):
        """
        파라미터 업데이트
        
        Args:
            params: 파라미터 리스트 [W1, b1, W2, b2, ...]
            grads: 기울기 리스트 [dW1, db1, dW2, db2, ...]
                  ← 이건 레이어의 backward()에서 계산된 것!
        """
        self.t += 1  # 시간 스텝 증가
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # 처음이면 모멘트 초기화
            if i not in self.m:
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)
            
            # 1차 모멘트 업데이트 (momentum)
            # m_t = β1 * m_{t-1} + (1 - β1) * g_t
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # 2차 모멘트 업데이트 (RMSprop)
            # v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction (초기 스텝에서 보정)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 파라미터 업데이트
            # θ_t = θ_{t-1} - lr * m_hat / (√v_hat + ε)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# 사용 예시
if __name__ == "__main__":
    # 1. Optimizer 생성
    optimizer = Adam(lr=0.001)
    
    # 2. 가상의 파라미터와 기울기
    W = np.random.randn(3, 2)
    b = np.zeros(2)
    
    dW = np.random.randn(3, 2)  # ← backward()에서 계산된 기울기
    db = np.random.randn(2)     # ← backward()에서 계산된 기울기
    
    print("업데이트 전 W:")
    print(W)
    
    # 3. 업데이트 (기울기를 받아서 사용)
    optimizer.update([W, b], [dW, db])
    
    print("\n업데이트 후 W:")
    print(W)
    