import cv2
import numpy as np
import pytesseract
from PIL import Image


# 显示图片
def cv_show(winname, image):
    cv2.imshow(winname, image)
    # 销毁窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 有些原图片的size不好处理，我们可以封装成一个函数来统一图片的size
# 封装resize功能.
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None  # 缩放后的宽和高
    (h, w) = image.shape[:2]
    # 不做处理
    if width is None and height is None:
        return image
    # 指定了resize的height
    if width is None:
        r = height / float(h)  # 缩放比例
        dim = (int(w * r), height)
    # 指定了resize的width
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# 进行透视变换.
# 透视变换要找到变换矩阵
# 变换矩阵要求原图的4个点坐标和变换之后的4个点的坐标
# 现在已经找到了原图的4个点的坐标。需要知道变换后的4个坐标
# 先对获取到的4个角点按照一定顺序（顺/逆时针）排序
# 排序功能是一个独立功能，可以封装成一个函数
def order_points(pts):
    # 创建全是0的矩阵, 来接收等下找出来的4个角的坐标.
    rect = np.zeros((4, 2), dtype='float32')
    # 列相加
    s = pts.sum(axis=1)
    # 左上的坐标一定是x,y加起来最小的坐标. 右下的坐标一定是x,y加起来最大的坐标.
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 右上角的x,y相减的差值一定是最小的.
    # 左下角的x,y相减的差值, 一定是最大.
    # diff的作用是后一列减前一列得到的差值
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 把透视变换功能封装成一个函数
def four_point_transform(image, pts):
    # 对输入的4个坐标排序
    rect = order_points(pts)
    # top_left简称tl，左上角
    # top_right简称tr，右上角
    # bottom_right简称br，右下角
    # bottom_left简称bl，左下角
    (tl, tr, br, bl) = rect
    # 空间中两点的距离，并且要取最大的距离确保全部文字都看得到
    widthA = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    widthB = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    max_width = max(int(widthA), int(widthB))
    heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    max_height = max(int(heightA), int(heightB))
    # 构造变换之后的对应坐标位置.
    dst = np.array([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]], dtype='float32')
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    # 透视变换
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


# 把图像预处理的功能封装成一个函数
def Image_Pretreatment(image,ratio,image_copy):

    # 图片预处理
    temimg = image.copy()
    # 灰度化处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv_show('gray',gray)
    # 高斯平滑
    Gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    # cv_show('Gaussian',Gaussian)
    # 边缘检测，寻找边界（为后续查找轮廓做准备）
    edged = cv2.Canny(Gaussian, 70, 200)
    # cv_show('edged',edged)
    # 查找轮廓
    cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    # 将轮廓按照面积降序排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    if(cv2.contourArea(cnts[0]) < temimg.shape[0]*temimg.shape[1]*0.7):
        warped_gray = cv2.cvtColor(temimg, cv2.COLOR_BGR2GRAY)
        ref = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # cv_show('ref', ref)
        return image_copy

    # 绘制所有轮廓
    image_contours = cv2.drawContours(image.copy(), cnts, -1, (0, 0, 255), 1)
    # cv_show('image_contours', image_contours)
    # 遍历轮廓找出最大的轮廓.
    for c in cnts:
        # 计算轮廓周长
        perimeter = cv2.arcLength(c, True)
        # 多边形逼近，得到近似的轮廓
        # 近似完后，只剩下四个顶点的角的坐标
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        # 最大的轮廓
        if len(approx) == 4:
            # 接收approx
            screen_cnt = approx
            break
    # 画出多边形逼近
    image_screen_cnt = cv2.drawContours(image.copy(), [screen_cnt], -1, (0, 0, 255), 1)
    # cv_show('image_screen_cnt', image_screen_cnt)
    # 进行仿射变换，使图片变正
    warped = four_point_transform(image_copy, screen_cnt.reshape(4, 2) * ratio)
    # cv_show('warped', warped)
    # 二值处理，先转成灰度图
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # 再二值化处理
    ref = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv_show('ref', warped)
    # # 旋转变正
    # dst = cv2.rotate(ref, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv_show('dst', dst)
    return ref



def rotate(img):
    image = img.copy()
    ratio = image.shape[0] / 700.0
    # 拷贝一份
    image_copy = image.copy()
    # 修改尺寸
    image = resize(image_copy, height=700)
    # cv_show('image', image)
    # 返回透视变换的结果
    ref = Image_Pretreatment(image,ratio,image_copy)

    return ref

#   if __name__ == "__main__":
#
#     # 读取图片
#     image = cv2.imread('./train/7.jpeg')
#     # 计算比例. 限定高度500
#     # 此时像素点都缩小了一定的比例，进行放射变换时要还原
#     ratio = image.shape[0] / 700.0
#     # 拷贝一份
#     image_copy = image.copy()
#     # 修改尺寸
#     image = resize(image_copy, height=700)
#     # cv_show('image', image)
#     # 返回透视变换的结果
#     ref = Image_Pretreatment(image)
#     # 把处理好的图片写入图片文件.
#     _ = cv2.imwrite('./scan.jpg', ref)
#     # pytesseract要求的image不是opencv读进来的image, 而是pillow这个包, 即PIL
#     text = pytesseract.image_to_string(Image.open('./scan.jpg'), lang='chi_sim+eng', config='--oem 1')
#     # 保存到本地
#     print(text)