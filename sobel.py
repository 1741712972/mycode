import cv2
import matplotlib.pyplot as plt
import numpy as np

def relate(img, H):         #直接求卷积结果
   #img是待求边缘的图像，H是相应卷积的算子
   new_img = img
   [a,b] = new_img.shape
   for i in range(1, a - 1):
       for j in range(1, b - 1):
           new_img[i, j] = (img[(i - 1):(i + 2), (j - 1):(j + 2)] * H).sum()
           #                     取i-1 ~ i+1 三个元素 即去掉了图像最边缘  .sum = sum(sum())
           new_img =np.array(new_img, dtype='uint8')
   return new_img


def sobel(img):
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    img_x = relate(img,sobel_x)
    img_y = relate(img,sobel_y)

    cv2.imshow("sobel_img_x", img_x)
    cv2.waitKey(0)

    cv2.imshow("sobel_img_y", img_y)
    cv2.waitKey(0)

    # img_xy = np.sqrt(img_x**2+img_y**2)

    img_xy = cv2.addWeighted(img_x,0.5,img_y,0.5,0)

    img_xy= np.array(img_xy, dtype='uint8')

    return img_xy



