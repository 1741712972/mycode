import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import test
import re
import os
#
# try:
#     from PIL import Image
#     from PIL import ImageDraw
#     from PIL import ImageFont
# except ImportError:
#     import Image

import gauss
import sobel




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def imgshow(imgname,img):
    cv2.imshow(imgname,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.

def xproject(img):
    x = np.sum(img,axis=0)

    # print(x)

    pltx = np.arange(img.shape[1])
    # plt.ylim(0, 2000)
    plt.plot(pltx, x)
    plt.title('xpro')
    plt.xlabel("pos value")
    plt.ylabel("number")
    plt.show()

    return x

def yproject(img):
    y = np.sum(img,axis=1)

    # print(y)

    plty = np.arange(img.shape[0])
    # plt.ylim(0, 2000)
    plt.plot(y, plty)
    plt.title('ypro')
    plt.xlabel("pos value")
    plt.ylabel("number")
    plt.show()

    return y

def drawcounters(img,drawimg):
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # 构建用于膨胀腐蚀的元素为1的卷积核
    # img = cv2.erode(img, kernel)  # 腐蚀
    # img =  cv2.Canny(img, 70, 200)

    counters, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawCountersimg = cv2.drawContours(drawimg, counters, -1, (40, 150, 200), 1)

    for i in range(len(counters)):
        x, y, w, h = cv2.boundingRect(counters[i])
        cv2.rectangle(drawimg, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 画出轮廓外接矩形

    imgshow('rec', drawimg)

def Imgprocess(img):
    #通常预处理阶段
    drawCountersimg = img.copy()
    drawrecimg = img.copy()
    draworginal = img.copy()
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # imgshow('gray',grayimg)
    #opencv_gauss = cv2.GaussianBlur(img, (3, 3), 0, 0)
    res, new_ostu_opencv = cv2.threshold(grayimg, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

    # imgshow('ostu',new_ostu_opencv)
    edged = cv2.Canny(new_ostu_opencv, 70, 200)
    # imgshow('canny',edged)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 构建用于膨胀腐蚀的元素为1的卷积核

    edged = cv2.dilate(edged, kernel,2)  # 膨胀
    # edged = cv2.erode(edged, kernel)  # 腐蚀

    counters,hier= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawCountersimg = cv2.drawContours(drawCountersimg, counters, -1, (40, 150, 200), 1)  # 第三个参数为负数则画出所有轮廓
    # imgshow('counters',drawCountersimg)

    for i in range(len(counters)):
        x, y, w, h = cv2.boundingRect(counters[i])
        cv2.rectangle(drawrecimg, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 画出轮廓外接矩形

    # imgshow('rec',drawrecimg)

    # xproject(edged)
    ypro = yproject(edged)
    pos = round(0.2*edged.shape[0])

    for i in range(ypro.shape[0]):
        if ypro[i] == max(ypro):
            pos = i
            break
    print(pos)

    new_img = edged[:][:pos+1]
    # imgshow('show', new_img)

    recogimg = draworginal[:][:pos+1]
    # imgshow('original', recogimg)

    return recogimg

def recongnize(img):

    img = img[:,round(0.7*img.shape[1]):]
    originalimg =img.copy()
    drawimg = img.copy()
    drawrecimg = img.copy()
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # imgshow('gray', grayimg)
    # opencv_gauss = cv2.GaussianBlur(img, (3, 3), 0, 0)
    res, new_ostu_opencv = cv2.threshold(grayimg, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    # imgshow('ostu', new_ostu_opencv)
    edged = cv2.Canny(new_ostu_opencv, 70, 150)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 构建用于膨胀腐蚀的元素为1的卷积核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # 构建用于膨胀腐蚀的元素为1的卷积核

    # edged = cv2.erode(edged,kernel_erode)

    edged = cv2.dilate(edged, kernel,10)  # 膨胀
    # edged = cv2.erode(edged, kernel_erode)

    xproject(edged)

    # imgshow('erode', edged)


    counters, hier = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawCountersimg = cv2.drawContours(drawimg, counters, -1, (40, 150, 200), 1)

    for i in range(len(counters)):
        x, y, w, h = cv2.boundingRect(counters[i])
        cv2.rectangle(drawimg, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 画出整体切分数据轮廓外接矩形

    # imgshow('rec',drawimg)

    counters = sorted(counters, key=cv2.contourArea, reverse=True)

    x, y, w, h = cv2.boundingRect(counters[1])
    cv2.rectangle(drawrecimg, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 画出riqi轮廓外接矩形
    img_riqi = originalimg[y:y+h+5,x:x+w+1]
    x, y, w, h = cv2.boundingRect(counters[2])
    cv2.rectangle(drawrecimg, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 画出daima轮廓外接矩形
    img_daima = originalimg[y:y+h+5,x:x+w+1]
    x, y, w, h = cv2.boundingRect(counters[3])
    cv2.rectangle(drawrecimg, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 画出haoma轮廓外接矩形
    img_haoma = originalimg[y:y+h+5,x:x+w+1]

    # imgshow('rec', drawrecimg)
    # imgshow('riqi', img_riqi)
    # imgshow('daima', img_daima)
    # imgshow('haoma', img_haoma)

    big_riqi = cv2.resize(img_riqi,(2*img_riqi.shape[1],2*img_riqi.shape[0]))
    big_daima = cv2.resize(img_daima, (2 * img_daima.shape[1], 2 * img_daima.shape[0]))
    big_haoma = cv2.resize(img_haoma, (2 * img_haoma.shape[1], 2 * img_haoma.shape[0]))

    # imgshow('big_riqi', big_riqi)
    # imgshow('big_daima', big_daima)
    # imgshow('big_haoma', big_haoma)


    grayimg_riqi = cv2.cvtColor(big_riqi, cv2.COLOR_BGRA2GRAY)
    # opencv_gauss = cv2.GaussianBlur(img, (3, 3), 0, 0)
    res, ostu_riqi = cv2.threshold(grayimg_riqi, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    grayimg_daima = cv2.cvtColor(big_daima, cv2.COLOR_BGRA2GRAY)
    # opencv_gauss = cv2.GaussianBlur(img, (3, 3), 0, 0)
    res, ostu_daima = cv2.threshold(grayimg_daima, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    grayimg_haoma = cv2.cvtColor(big_haoma, cv2.COLOR_BGRA2GRAY)
    # opencv_gauss = cv2.GaussianBlur(img, (3, 3), 0, 0)
    res, ostu_haoma = cv2.threshold(grayimg_haoma, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    riqi = pytesseract.image_to_string(img_riqi, lang='chi_sim+eng')
    daima = pytesseract.image_to_string(img_daima, lang='chi_sim+eng')
    haoma = pytesseract.image_to_string(img_haoma, lang='chi_sim+eng')


    # print('日期是:', riqi,'格式:',type(riqi))
    # print('代码是:', daima)
    # print('号码是:', haoma)

    result = finddata([riqi,daima,haoma])

    return result


def finddata(data):
    riqi = ''
    daima = ''
    haoma = ''

    list_riqi_year = re.findall(r'[0-9]{4}', data[0])
    if(len(list_riqi_year)>=1):
        riqi_year = re.findall(r'[0-9]{4}', data[0])[0]
    else:
        riqi_year = ''

    list_riqi_month = re.findall(r'[0-9]{2}', data[0])
    if (len(list_riqi_month) >= 3):
        riqi_month = re.findall(r'[0-9]{2}', data[0])[2]
    else:
        riqi_month = ''

    list_riqi_day = re.findall(r'[0-9]{2}', data[0])
    if (len(list_riqi_day) >= 4):
        riqi_day = re.findall(r'[0-9]{2}', data[0])[3]
    else:
        riqi_day = ''

    riqi = riqi_year+riqi_month+riqi_day


    list_daima = re.findall(r'[0-9]{12}', data[1])
    if(len(list_daima)>=1):
        daima = re.findall(r'[0-9]{12}', data[1])[0]
    else:
        daima = ''


    list_haoma = re.findall(r'[0-9]{8}', data[2])
    if(len(list_haoma)>=1):
        haoma = re.findall(r'[0-9]{8}', data[2])[0]
    else:
        haoma = ''



    return [daima,haoma,riqi]

    #通过腐蚀操作进行投影
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))  # 构建用于膨胀腐蚀的元素为1的卷积核
    # ostu_haoma = cv2.erode(ostu_haoma, kernel)  # 腐蚀
    # imgshow('erode_haoma',ostu_haoma)
    # xproject(ostu_haoma)
    # ostu_daima = cv2.erode(ostu_daima, kernel)  # 腐蚀
    # imgshow('erode_daima',ostu_daima)
    # xproject(ostu_daima)
    # ostu_riqi = cv2.erode(ostu_riqi, kernel)  # 腐蚀
    # imgshow('erode_riqi',ostu_riqi)
    # xproject(ostu_riqi)


    # drawcounters(ostu_riqi,big_riqi)
    # drawcounters(ostu_daima, big_daima)
    # drawcounters(ostu_haoma, big_haoma)


    # text = pytesseract.image_to_string(img, lang='chi_sim+eng')
    # print(text)

def judgestring(str1,str2):   #返回两个字符串能够匹配的个数

    if(str1=='' or str2==''):
        return 0
    elif(str1 == str2): #返回正确字符数
        return len(str1)
    else:
        correct_num = 0
        length = len(str1)
        if(len(str1)!=len(str2)):
            length = min(len(str1),len(str2))#遍历之前 先对遍历长度进行修正 以长度短的为准

        for i in range(length):
            if(str1[i] == str2[i]):
                correct_num  = correct_num +1 #计算匹配字符的个数

        return correct_num




def judgerate(result,myresult):

    num_img = len(result)*len(result[0])
    sum_character = 0
    correct_character = 0
    correct_num = 0

    if(len(result)!=len(myresult)):
        print('识别出错')
    for i in range(len(result)):   #i是第n张图片

        for j in range(len(result[i])): #j是第n个要比较的字符串
            sum_character += len(result[i][j])
            correct_character += judgestring(result[i][j],myresult[i][j])
            if(judgestring(result[i][j],myresult[i][j]) == len(result[i][j])):
                correct_num = correct_num + 1

        print('第', i+1, '张图片的标准结果为', result[i])
        print('第', i+1, '张图片的识别结果为', myresult[i])

    print('每条信息识别准确率为',correct_num/num_img*100)
    print('字符识别准确率为', correct_character / sum_character * 100)



    return 0




if __name__ == '__main__':



    # print_hi('PyCharm')
    # img = image_transgray('1.jpg')
    # gray_img = calculategray(img)
    # gauss_img = my_gauss(gray_img)
    # sobel_img = my_sobel(gauss_img)
    #
    # featureextract.getfeature(gray_img,cv2.imread('1.jpg'))


    dir  = './data/'
    filelist = os.listdir(dir)

    resfile = filelist[len(filelist)-1]#初始化读取文件数组
    for file in filelist:
        if(file.endswith('.txt')):
            resfile = file
            filelist.remove(file)#找到后把这个txt删除，只剩下原来的数组
            break
    resfile = dir+resfile #结果文件
    print(resfile)
    filelist.sort()#待识别图像名称aaaaaa
    print(filelist)

    with open(resfile) as file: #读取识别标准结果
        content = file.readlines()
    print(content)
    results = [res.split() for res in content] #标准结果数组aaaaaa
    print(results)


    dstres = []

    for file in filelist:
        print(file)
        img = cv2.imread(dir+file)
        try:
            imgshow('example', img)
        except cv2.error:
            dstres.append(['', '', ''])
            print(file, '不能用cv2读取')
            continue

        # imgshow('example', img)
        img = test.rotate(img)
        # imgshow('rotateresult', img)
        processedimg = Imgprocess(img)
        # imgshow('processedimg', processedimg)
        res = recongnize(processedimg)
        dstres.append(res)

        # if(img == None):
        #     dstres.append(['','',''])
        #     print(file,'不能用cv2读取')
        #     continue
        #


    print(dstres)
    print(len(dstres))

    judgerate(results,dstres)


    # img = cv2.imread('./data/011002200411.jpeg')
    # imgshow('example', img)
    # img = test.rotate(img)
    # imgshow('rotateresult', img)
    # processedimg = Imgprocess(img)
    # imgshow('processedimg', processedimg)
    # rr = recongnize(processedimg)
    #
    # print(rr)


    # img = cv2.imread('./train/10.jpeg')
    # if(img == None):
    #     print('该图像不可通过CV读取')







