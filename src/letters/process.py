"""
答题卡裁剪，单个字符分割 process.py
"""
import copy
import os
import cv2
from imutils import contours
import shutil
import numpy as np
from PIL import Image


def deblur(image):
    """
    高斯模糊的逆操作
    :param image:
    :return:
    """
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.addWeighted(image, 1.5, blur, -0.5, 0)


def sharpen(image):
    """
    锐化
    :param image:
    :return:
    """
    kerneli = np.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]])
    return cv2.filter2D(image, -1, kerneli)


def delet_contours(contour, delete_list):
    """
    删除指定轮廓列表中的轮廓
    :param contour:
    :param delete_list:
    :return:
    """
    delta = 0
    for i in range(len(delete_list)):
        # print("i= ", i)
        del contour[delete_list[i] - delta]
        delta = delta + 1
    return contour


def answercrop():
    """
    从答题卡上裁剪选择题
    :return:
    """
    # 单通道读取答题卡
    img = cv2.imread('uploads/answer.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (6000, 8000))
    # 二值化
    _, binary_img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        # 获取坐标和长宽
        x, y, width, height = cv2.boundingRect(contour)
        ar = width / float(height)
        # 筛选合适轮廓
        if 2.3 <= ar <= 2.7 and 3000 <= width <= 4200 and 1000 <= height <= 1800:
            # 对裁剪区域留50像素，方便分割时查找轮廓
            table_img = img[y - 50:y + height + 50, x - 50:x + width + 50]
            table_img = cv2.resize(table_img, (1150, 500), interpolation=cv2.INTER_AREA)
            cv2.imwrite('uploads/answer_form.png', table_img)


def sortc(cnts):
    """
    对轮廓列表进行排序
    :param cnts: 
    :return: 
    """
    sorted_cnts1 = []
    sorted_cnts2 = []
    for cnt in cnts:
        # 计算轮廓的矩
        M = cv2.moments(cnt)
        # 计算轮廓的质心坐标
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # 获取第11-20题答案的轮廓
        if 410 < cy < 430:
            sorted_cnts2.append(cnt)
        # 获取第1-10题答案的轮廓
        if 180 < cy < 200:
            sorted_cnts1.append(cnt)
    # sorted_cnts1 = sorted(sorted_cnts1, key=lambda x: x[0][0][0])
    # sorted_cnts2 = sorted(sorted_cnts2, key=lambda x: x[0][0][0])
    # 根据坐标从左到右排序
    sorted_cnts1, sorted_cnts2 = contours.sort_contours(sorted_cnts1, method="left-to-right")[0], \
        contours.sort_contours(sorted_cnts2, method="left-to-right")[0]
    return sorted_cnts1, sorted_cnts2


def coo(contour, img):
    """
    根据轮廓坐标截取图像
    :param contour:
    :param img:
    :return:
    """
    x, y, width, height = cv2.boundingRect(contour)
    table_img = img[y + 5:y + height - 5, x + 5:x + width - 5]
    table_img = cv2.resize(table_img, (280, 280), interpolation=cv2.INTER_CUBIC)
    return table_img


# def split():
#     """
#     分割答题卡答案表格为单个字符
#     :return:
#     """
#     img = cv2.imread('uploads/answer_form.png')
#     # 灰度
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # 二值化
#     _, binary_img = cv2.threshold(gray_img, 165, 255, cv2.THRESH_BINARY)
#     cv2.imshow('binary', binary_img)
#     cv2.waitKey(0)
#     # 轮廓筛选排序
#     cnts, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = [x for x in cnts if 15000 > cv2.contourArea(x) > 11000]
#     # 排序
#     sorted_cnts1, sorted_cnts2 = sortc(cnts)
#     image_dir = '../../english_images/'
#     # 清空文件夹
#     shutil.rmtree(image_dir)
#     os.mkdir(image_dir)
#     for i, contour in enumerate(sorted_cnts1):
#         table_img = coo(contour, img)
#         cv2.imwrite(image_dir + str(i + 1) + '.png', table_img)
#     for i, contour in enumerate(sorted_cnts2):
#         table_img = coo(contour, img)
#         cv2.imwrite(image_dir + str(i + 11) + '.png', table_img)


def split():
    """
    分割答题卡为单个字符
    :return:
    """
    image = cv2.imread('uploads/answer_form.png', 1)
    # 灰度图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    # binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
    ret, binary = cv2.threshold(~gray, 127, 255, cv2.THRESH_BINARY)
    # 识别横线，通过调整scale大小可获取想要的横线长度
    rows, cols = binary.shape
    scale = 20
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    # 腐蚀
    eroded = cv2.erode(binary, kernel, iterations=1)
    # 膨胀
    dilatedcol = cv2.dilate(eroded, kernel, iterations=1)
    # 识别竖线，原理同上
    scale = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)
    # 标识交点
    bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)
    # cv2.imwrite("my.png",bitwiseAnd) #将二值像素点生成图片保存
    # 标识表格
    merge = cv2.add(dilatedcol, dilatedrow)
    # 两张图片进行减法运算，去掉表格框线
    merge2 = cv2.subtract(binary, merge)
    # 识别黑白图中的白色交叉点，将横纵坐标取出
    ys, xs = np.where(bitwiseAnd > 0)

    mylisty = []  # 纵坐标
    mylistx = []  # 横坐标

    # 通过排序，获取跳变的x和y的值，说明是交点，否则交点会有好多像素值值相近，我只取相近值的最后一点
    # 这个10的跳变不是固定的，根据不同的图片会有微调，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
    i = 0
    myxs = np.sort(xs)
    for i in range(len(myxs) - 1):
        if myxs[i + 1] - myxs[i] > 10:
            mylistx.append(myxs[i])
        i = i + 1
    mylistx.append(myxs[i])  # 要将最后一个点加入

    i = 0
    myys = np.sort(ys)
    for i in range(len(myys) - 1):
        if myys[i + 1] - myys[i] > 10:
            mylisty.append(myys[i])
        i = i + 1
    mylisty.append(myys[i])  # 要将最后一个点加入

    # 清空文件夹
    image_dir = '../../english_images/'
    shutil.rmtree(image_dir)
    os.mkdir(image_dir)
    # x，y坐标分割表格
    count = 1
    for j in range(len(mylistx) - 1):
        # 在分割时，第一个参数为y坐标，第二个参数为x坐标
        ROI = image[mylisty[1] + 3:mylisty[2] - 3, mylistx[j]:mylistx[j + 1] - 3]  # 减去3的原因是由于缩小ROI范围
        cv2.imwrite(image_dir + str(count) + '.png', ROI)
        ROI = image[mylisty[3] + 3:mylisty[4] - 3, mylistx[j]:mylistx[j + 1] - 3]  # 减去3的原因是由于缩小ROI范围
        cv2.imwrite(image_dir + str(count + 10) + '.png', ROI)
        count += 1
