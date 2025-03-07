"""
可视化展示 visualize.py
"""
import cv2
from PIL import Image
import matplotlib as mpl
from matplotlib import pyplot as plt
from src.utils.load_emnist import *


def get_mapping(num, with_type='letters'):
    """
    根据 mapping，由传入的 num 计算 UTF8 字符。
    :param num:
    :param with_type:
    :return:
    """
    if with_type == 'byclass':
        if num <= 9:
            return chr(num + 48)  # 数字
        elif num <= 35:
            return chr(num + 55)  # 大写字母
        else:
            return chr(num + 61)  # 小写字母
    elif with_type == 'letters':
        return chr(num + 64)  # 大写字母
    elif with_type == 'digits':
        return chr(num + 96)
    else:
        return num


def show_pred_result(path, result):
    """
    可视化的方式展示图片和预测结果
    :param path: 图片路径
    :param result: 预测结果
    :return: None
    """
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体支持
    img = Image.open(path)
    plt.figure(path)  # 文件路径作窗口标题
    plt.axis('off')  # 关闭坐标系显示
    plt.title("预测结果是：" + result)
    plt.imshow(img)
    plt.show()


def show_data_set(images_data, num=1000):
    """
    转换数据集中的内容并在窗口中展示
    :param images_data:
    :param num: 展示几个
    :return: None
    """
    images = []
    img = []

    for i in range(images_data.shape[0]):
        im = images_data[i]
        im = im.reshape(28, 28)
        if len(img) == 0:
            img = im
        else:
            # 横向组合
            img = np.hstack((img, im))

        # 每行显示30个数字图片
        if img.shape[1] / 28 == 45:
            if len(images) == 0:
                images = img.copy()
                img = []
            else:
                # 纵向组合
                images = np.vstack((images, img))
                img = []

        # 显示前 num 个
        if i == num:
            break

    cv2.imshow('data_set_visualization', images)
    cv2.waitKey()
    cv2.destroyAllWindows()
