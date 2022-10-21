# This is a sample Python script.
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import PIL

import gauss
import sobel
import featureextract
import pandas as pd
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from cv2.mat_wrapper import Mat

# np_c 矩阵列拓展，行不变，列增加
# np_r 矩阵行拓展，列不变，行增加
def image_transgray(path):
    img = cv2.imread(path)
    cv2.imshow("test", img)
    cv2.waitKey(0)

    height = img.shape[0]
    width = img.shape[1]
    channel = img.shape[2]

    mygrayimg = np.zeros((height,width),dtype=np.uint8)

    for i in range(height): #按照RGB的比例进行灰度值转换
        for j in range(width):
            mygrayimg[i,j] = (img[i,j,0]*30 + img[i,j,1]*59 + img[i,j,2]*11)/100


    img = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    print(img[20,30])

    cv2.imshow("mygray", mygrayimg)
    cv2.waitKey(0)

    cv2.imshow("test",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img






# def myostu_test(gray):
#     num = sum(gray[x] for x in range(256))#总像素点数
#     print(num)
#
#     temmartix = gray
#     print(temmartix)
#
#     #归一化直方图
#     for i in range(len(gray)):
#         gray[i] = gray[i]/num
#
#
#     #依次计算每个灰度值作为阈值前景像素与背景像素的方差，来确定最佳阈值
#     tem = -1 #临时用来存g的变量
#     res = 0 #最终确定的阈值
#     w0 = 0 #前景像素点数占图像的比例
#     u0 = 0 #前景像素的平均灰度
#     w1 = 1 - w0 #背景像素点数占图像的比例
#     u1 = 0 #背景像素的平均灰度
#     preamount = 1  #前景像素点数
#     backamount = num - preamount #背景像素点数
#     pregray = 0  #前景灰度值和
#     backgary = 0 #背景灰度值和
#     sumgray = sum(temmartix[x]*x for x in range(256)) #灰度值总数
#
#     for j in range(256):
#         for i in range(j):
#             preamount = preamount +  temmartix[i]
#         print(preamount)
#         # if(preamount == 0):
#         #     preamount = 1
#         backamount = num - preamount
#         print(backamount)
#         # if (backamount == 0):
#         #     backamount = 1
#
#         w0 = preamount/num
#         w1 = backamount/num
#         pregray = sum(temmartix[x]*x for x in range(j))
#         backgary = sumgray - pregray
#         u0 = pregray/preamount
#         u1 = backgary/backamount
#         g = w0 * w1 * (u1 - u0)**2
#         if g > tem:
#            tem = g
#            print(g)
#            res = j
#         # w0 = sum(gray[x] for x in range(j)) #因为进行了归一化，所以所占比例就是值求和
#         # w1 = 1-w0
#         # u0 = sum(temmartix[x]*x for x in range(j))/sum(temmartix[x] for x in range(j))
#         # u1 = sum(temmartix[x] * x for x in range(j+1,255))/sum(temmartix[x] for x in range(j+1,255))
#         # g = w0*w1*(u0-u1)*(u0-u1)
#         # if g > tem:
#         #     tem = g
#         #     print(g)
#         #     res = j
#         #     print(j)
#     print(res)
#
#     return res
#
#
#
#
#



def myostu(gray):
    num = sum(gray[x] for x in range(256))  # 总像素点数
    print(num)
    temmartix = gray
    print(temmartix)

    sumgray = sum(temmartix[x] * x for x in range(256))  # 灰度值总数
    temvalue = -1#用来存g最大类间方差
    res = 0 #用来存阈值

    for j in range(256):
        print(j)
        preamount = 1 #前景像素点数
        backamount = 0 #背景像素点数
        pregray = 0 #前景灰度值和
        backgray = 0 #背景灰度值和

        for i in range(j):
            preamount = preamount + temmartix[i]#计算前景像素点数
        print(preamount)
        backamount = num - preamount #背景像素点数

        w0 = preamount/num
        w1 = backamount/num

        pregray = sum(temmartix[x]*x for x in range(j))
        backgary = sumgray - pregray

        u0 = pregray / preamount
        u1 = backgary / backamount

        g = w0 * w1 * (u1 - u0) ** 2

        if g > temvalue:
            temvalue = g
            print(g)
            res = j
    print(res)

    return res

def calculategray(img):
    x,y = img.shape[:2]#获取图像大小
    gray = [x*0 for x in range(256)]#初始化直方图数组
    print(len(gray))
    print(gray)
    for i in range(x):
        for j in range(y):
            gray[img[i][j]] = gray[img[i][j]]+1
    print(gray)
    print(sum(gray[x] for x in range(256)))
    pltx = np.arange(256)
    plt.ylim(0,2000)
    plt.plot(pltx,gray)
    plt.xlabel("gray value")
    plt.ylabel("number")
    plt.show()
    #根据经验结果确定了阈值为200
    res,new_img = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
    #使用opencv中的ostu算法结果
    res,new_ostu_opencv = cv2.threshold(img,0,255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    #自己编写的ostu算法
    myvalue = myostu(gray)
    res,new_ostu_myostu = cv2.threshold(img,myvalue,255,cv2.THRESH_BINARY)
    #自己编写的ostu的结果
    cv2.imshow("my_ostu_result", new_ostu_myostu)
    # cv2.waitKey(0)
    #调用的opencv函数的结果
    cv2.imshow("opencv_ostu_result", new_ostu_opencv)
    # cv2.waitKey(0)
    #根据经验设置阈值结果
    cv2.imshow("experinece_result", new_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return new_ostu_opencv

def my_gauss(img):
    my_gauss_img = gauss.fil(img)
    my_gauss_img = np.array(my_gauss_img, dtype='uint8')

    opencv_gauss = cv2.GaussianBlur(img,(3,3),0,0)

    cv2.imshow("my_gauss_result", my_gauss_img)
    cv2.waitKey(0)

    cv2.imshow("opencv_gauss_result", opencv_gauss)
    cv2.waitKey(0)

    return opencv_gauss


def my_sobel(img):
    my_sobel_img = sobel.sobel(img)
    my_sobel_img = np.array(my_sobel_img, dtype='uint8')#此步为了防止超范围

    opencv_sobel_x = cv2.Sobel(img,-1,1,0,ksize=3)
    opencv_sobel_y = cv2.Sobel(img,-1,0,1, ksize=3)
    opencv_sobel_xy = cv2.addWeighted(opencv_sobel_x,0.5,opencv_sobel_y,0.5,0)

    cv2.imshow("opencv_sobel_result", opencv_sobel_xy)
    # cv2.waitKey(0)

    cv2.imshow("my_sobel_result",my_sobel_img)
    cv2.waitKey(0)

    return 0



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # print_hi('PyCharm')
    img = image_transgray('1.jpg')
    gray_img = calculategray(img)
    # gauss_img = my_gauss(gray_img)
    # sobel_img = my_sobel(gauss_img)
    #
    featureextract.getfeature(gray_img,cv2.imread('1.jpg'))
    # text = pytesseract.image_to_string(PIL.Image.open('2.jpg'), lang='chi_sim')
    # print(text)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
