import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    2차원 배열을 입력받아 다시 다수의 이미지 묶음으로 변환한다.
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
def softmax(y_pred):
    # 입력이 (batch_size, num_classes)라고 가정
    
    # 1. 오버플로우 방지를 위해 각 샘플의 최댓값을 뺍니다.
    c = np.max(y_pred, axis=1, keepdims=True)
    
    # 2. 최댓값을 뺀 값으로 exp를 계산합니다. (가장 큰 값이 0이 되므로 exp 결과는 최대 1)
    z = np.exp(y_pred - c)
    # 차원이 1차원인 경우에는 axis 를 제거
    if z.ndim == 1:
        t = np.sum(z)
    else:
        t = np.sum(z, axis=1, keepdims=True)
    
    return z / t

if __name__ == "__main__":
    # im2col 테스트
    print("=== im2col 테스트 ===")
    
    # 간단한 예제: 1개 이미지, 1채널, 4x4 크기
    test_img = np.arange(1, 17).reshape(1, 1, 4, 4)
    print("입력 이미지 (1, 1, 4, 4):")
    print(test_img[0, 0])
    print()
    
    # 3x3 필터, stride=1, pad=0
    col = im2col(test_img, filter_h=3, filter_w=3, stride=1, pad=0)
    print("im2col 결과 (3x3 필터, stride=1, pad=0):")
    print(f"Shape: {col.shape}")  # (4, 9) 예상
    print(col)
    print()
    
    # 2x2 필터, stride=2, pad=1
    col2 = im2col(test_img, filter_h=2, filter_w=2, stride=2, pad=1)
    print("im2col 결과 (2x2 필터, stride=2, pad=1):")
    print(f"Shape: {col2.shape}")
    print(col2)
    print()
    
    # col2im 테스트
    print("=== col2im 테스트 (역변환) ===")
    reconstructed = col2im(col, input_shape=(1, 1, 4, 4), filter_h=3, filter_w=3, stride=1, pad=0)
    print("복원된 이미지 shape:", reconstructed.shape)
    print("복원 확인 (겹치는 영역은 합산됨):")
    print(reconstructed[0, 0])


    
