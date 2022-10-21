import cv2
import matplotlib.pyplot as plt
import numpy as np

def getfeature(grayimg,img):
    temimg = img.copy()
    rec1 = img.copy()
    rec2 = img.copy()


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,3))#构建用于膨胀腐蚀的元素为1的卷积核


    grayimg = cv2.dilate(grayimg, kernel)  # 膨胀
    grayimg = cv2.erode(grayimg, kernel)  # 腐蚀


    counters,hierarchy = cv2.findContours(grayimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #RETR_EXTERNAL 外部轮廓 LIST 所有轮廓并存到链表 CCOMP 存为两层 空洞与外部
                                            #TREE 所有轮廓并重构嵌套轮廓层次（常用） CHAIN_APPROX_SIMPLE 压缩水平垂直与斜的部分
    newimg = cv2.drawContours(temimg,counters,-1,(40,150,200),1)#第三个参数为负数则画出所有轮廓

    cv2.imshow("counters all",newimg)
    cv2.waitKey(0)

    area = np.zeros(len(counters)) #每个轮廓的面积
    lens = np.zeros(len(counters)) #每个轮廓的周长
    cx = np.zeros(len(counters)) #每个轮廓的重心的X坐标
    cy = np.zeros(len(counters)) #每个轮廓的重心的Y坐标
    count = 0
    print("轮廓数量为：")
    print(len(counters))

    for i in range(len(counters)):
        area[i] = round(cv2.contourArea(counters[i]))#计算每个轮廓的面积
        lens[i] = round(cv2.arcLength(counters[i],True))#计算每个轮廓周长

        M = cv2.moments(counters[i])#获取轮廓的矩 用来求重心
        if(M['m00']!= 0):
            cx[i] = M['m10'] / M['m00']  # 求重心的X坐标
            cy[i] = M['m01'] / M['m00']  # 求重心的Y坐标
        else :
            print("除0异常")

        if(cx[i]/img.shape[1] >= 0.6):
            count = count + 1
            x, y, w, h = cv2.boundingRect(counters[i])  # 计算轮廓的外接矩形
            cv2.rectangle(rec1, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 画出轮廓外接矩形
            print("第",count,"个字符的重心为:","(",cx[i] ,",", cy[i],")")
            print("第", count, "个字符的面积为:", area[i])
            print("第", count, "个字符的轮廓长度为:", lens[i])

            minrect = cv2.minAreaRect(counters[i])
            minbox = np.int0(cv2.boxPoints(minrect))
            cv2.drawContours(rec2, [minbox], 0, (255, 0, 0), 1)




    cv2.imshow("rec1", rec1)
    cv2.waitKey(0)

    cv2.imshow("rec2", rec2)
    cv2.waitKey(0)

    print(area)
    print(lens)
    print(cx)
    print(cy)
    return 0