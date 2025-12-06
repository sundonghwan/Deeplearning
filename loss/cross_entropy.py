import numpy as np
from utils.utils import softmax

def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Loss
    
    Args:
        y_pred: 예측값 (N, C) - softmax 출력
        y_true: 정답 레이블 (N, C) - one-hot 또는 (N,) - 클래스 인덱스
    
    Returns:
        loss: 스칼라 값
    """
    batch_size = y_pred.shape[0]
    
    # 수치 안정성을 위해 작은 값 추가 (log(0) 방지)
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # one-hot 벡터인 경우
    if y_true.ndim == 2:
        return -np.sum(y_true * np.log(y_pred)) / batch_size
    
    # 클래스 인덱스인 경우 (더 효율적)
    else:
        return -np.sum(np.log(y_pred[np.arange(batch_size), y_true])) / batch_size

def softmax_cross_entropy_loss(logits, y_true):
    """
    Softmax + Categorical Cross Entropy Loss (수치 안정적)
    
    Args:
        logits: 예측값 (N, C) - softmax 적용 전 로짓
        y_true: 정답 레이블 (N,) - 클래스 인덱스 또는 (N, C) - one-hot
    
    Returns:
        loss: 스칼라 값
    """
    batch_size = logits.shape[0]
    
    # 1. 수치 안정성을 위해 최댓값 빼기
    logits_max = np.max(logits, axis=1, keepdims=True)
    logits_shifted = logits - logits_max
    
    # 2. Log-sum-exp 트릭으로 안정적 계산
    # log(softmax(x)) = x - log(sum(exp(x)))
    log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=1, keepdims=True))
    log_probs = logits_shifted - log_sum_exp
    
    # 3. Cross entropy 계산
    if y_true.ndim == 1:
        # 클래스 인덱스인 경우
        correct_log_probs = log_probs[np.arange(batch_size), y_true]
        loss = -np.sum(correct_log_probs) / batch_size
    else:
        # one-hot 인코딩인 경우
        loss = -np.sum(y_true * log_probs) / batch_size
    
    return loss    