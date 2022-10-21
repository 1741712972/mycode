# filter.py
import cv2
import numpy as np
import math



# 计算模板的大小以及模板
def compute(delta):
    k = round(3 * delta) * 2 + 1 #根据高斯分布函数，大部分的相关像素点都分布在（-3d~3d）这里的d指的是高斯函数里的方差，取整计算需要的模板大小
    print('模的矩阵大小:', k)      #模板大小
    H = np.zeros((k, k))         #初始化高斯卷积核
    k1 = (k - 1) / 2             #确定卷积核的中心位置
    for i in range(k):
        for j in range(k):
            H[i, j] = (1 / (2 * 3.14 * (delta ** 2))) * math.exp((-(i - k1) ** 2 - (j - k1) ** 2) / (2 * delta ** 2))
    k3 = [k, H]
    print(H)
    print(sum(sum(H)))           #计算卷积核的所有元素和
    return k3


# 相关
def relate(a, b, k):         #计算卷积的过程（对应元素乘积求和）
    n = 0
    # print(a)
    # print(b)
    sum1 = np.zeros((k, k))

    for m in range(k):
        for n in range(k):
            sum1[m, n] = a[m, n] * b[m, n]
    return sum(sum(sum1))


# 高斯滤波
def fil(imag, delta=0.7):
    k3 = compute(delta)
    k = k3[0]  #模板大小
    H = k3[1]  #模板矩阵
    k1 = (k - 1) / 2
    [a, b] = imag.shape
    k1 = int(k1)
    newrow = np.zeros((k1, b))
    newraw = np.zeros(((a + (k - 1)), k1))  #               0,0,0,0,0
    imag1 = np.r_[newrow,imag] #矩阵上面扩充行                 0,1,1,1,0
    imag1 = np.r_[imag1,newrow] #矩阵下面扩充行                0,1,1,1,0
    imag1 = np.c_[newraw,imag1] #矩阵左边扩充列                0,1,1,1,0
    imag1 = np.c_[imag1, newraw] #矩阵右边扩充列               0,0,0,0,0
    y = np.zeros((a, b))
    sum2 = sum(sum(H))
    for i in range(k1, (k1 + a)):   #k1是指从原图有数据的像素点开始，略过填充的0元素列，因此下面计算结果时需要减掉扩充的0元素列
        for j in range(k1, (k1 + b)):
            y[(i - k1), (j - k1)] = relate(imag1[(i - k1):(i + k1 + 1), (j - k1):(j + k1 + 1)], H, k) / sum2 #除以矩阵元素和
            #                                               (k-1)/2+1（不包括最后那个点）
    return y
