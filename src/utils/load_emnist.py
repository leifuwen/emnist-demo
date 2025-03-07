"""
加载数据集 load_emnist.py
"""
import os
import cv2
import numpy as np
import gzip
from PIL import Image
from matplotlib import pyplot as plt


# from src.letters.letter_predict import prepare_image


def image_reverse(input_image):
    """
    图像反色
    :param input_image: 原图像
    :return: 反色后的图像
    """
    # 输入图像的副本
    input_image_cp = np.copy(input_image)
    # 输入图像像素的最大值
    pixels_value_max = np.max(input_image_cp)
    # 输出图像
    output_imgae = pixels_value_max - input_image_cp

    return output_imgae


def read_idx3(filename):
    """
    读取gz格式的数据集图像部分，并返回

    :param filename: extension name of the file is '.gz'
    :return: images data, shape -> num, rows, cols
    """
    with gzip.open(filename) as fo:
        print('Reading images...')
        buf = fo.read()

        offset = 0  # 偏移量
        # 首先获取的是这个数据集的头部数据，通常是元数据。
        #   '>i'  表示顺序读取，并且数据类型为整数
        #   4  读4个单位
        #   offset 偏移量
        # 返回的是一个数组，赋值给header
        header = np.frombuffer(buf, dtype='>i', count=4, offset=offset)
        print(header)
        magic_number, num_images, num_rows, num_cols = header
        # magic number 即幻数，意义不明，只是读取时需要占位所以声明了
        print("\tmagic number: {}, number of images: {}, number of rows: {}, number of columns: {}" \
              .format(magic_number, num_images, num_rows, num_cols))
        # 计算偏移量，以读取后续的内容
        # size = 数组长度
        # itemsize = 每个元素的大小
        # 因此乘起来就是跳过header的内容，读后续的内容
        offset += header.size * header.itemsize
        # 读取真正的数据。>B 表示是二进制数据
        data = np.frombuffer(buf, '>B', num_images * num_rows * num_cols, offset).reshape(
            (num_images, num_rows, num_cols))
        # .reshape 表示按传入的参数重新构造这个数组
        data_list = []
        for x in data:
            data_list.append(image_reverse(x))
        data = np.array(data_list)
        return data, num_images


def read_idx1(filename):
    """
    读取gz格式的数据集标签部分，并返回

    :param filename: extension name of the file is '.gz'
    :return: labels
    """
    with gzip.open(filename) as fo:
        print('Reading labels...')
        buf = fo.read()

        offset = 0
        header = np.frombuffer(buf, '>i', 2, offset)
        magic_number, num_labels = header
        print("\tmagic number: {}, number of labels: {}" \
              .format(magic_number, num_labels))

        offset += header.size * header.itemsize

        data = np.frombuffer(buf, '>B', num_labels, offset)
        return data


def read_img(listdirname):
    """
    读取文件夹下的所有图片，并返回numpy数据
    :param listdirname: 文件夹路径
    :return: 图片的numpy数组，图片数量
    """
    # 获取文件夹下的图片
    files = os.listdir(listdirname)
    # 分割图片名并按图片名前的数字排序
    files.sort(key=lambda x: x.split('.')[0])
    img_list = []
    # 将图片加入列表
    for filename in files:
        img = cv2.imread(listdirname + "/" + filename, cv2.IMREAD_GRAYSCALE)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # 将图片放大
        img = cv2.resize(img, (28, 28))[5:23, 5:23]
        img = cv2.resize(img, (28, 28))
        for i in range(10):
            img_list.append(img)
    # 将列表转换成np数组
    data = np.array(img_list, '>B').reshape(len(img_list), 28, 28)
    return data, len(img_list)


def read_img_lable(listdirname):
    """
    读取txt格式标签并返回np数组
    :param listdirname:文件路径
    :return:
    """
    data_list = []
    for x in np.loadtxt(listdirname, '>B'):
        for i in range(10):
            data_list.append(x)
    data = np.array(data_list, '>B')
    return data


def read_csv_lable(listdirname):
    """
    读取csv格式标签并返回np数组
    :param listdirname:文件路径
    :return:
    """
    data_list = []
    # noinspection PyTypeChecker
    for x in np.genfromtxt(listdirname, dtype='>B', delimiter=',', skip_header=1, usecols=1):
        for i in range(10):
            data_list.append(x)
    data = np.array(data_list, '>B')
    return data


def makedataset(listdirname):
    # 获取文件夹下的图片
    files = os.listdir(listdirname)
    # 分割图片名并按图片名前的数字排序
    files.sort(key=lambda x: x.split('.')[0])
    fcount = int(files[-1].split('.')[0])
    imgs = os.listdir('../../english_images/')
    # 分割图片名并按图片名前的数字排序
    imgs.sort(key=lambda x: int(x.split('.')[0]))
    labels = [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4]
    with open('./../../data_set_self_letters/label.txt', 'a') as f:
        for i in labels:
            f.write(" " + str(i))
        f.close()
    for imgname in imgs:
        fcount += 1
        img = cv2.imread('../../english_images/' + imgname)
        cv2.imwrite(f'./../../data_set_self_letters/img/{fcount:04}.png', img)


# makedataset('./../../data_set_self_letters/img')


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

# dataa, enu = read_img("./../../data_set_self_letters/img")
# read_csv_lable('./../../data_set_self_letters/english.csv')
# read_img_lable("./../../data_set_self_letters/label.txt")
# dataa, num = read_idx3('./../../data_set_emnist_letters/emnist-letters-train-images-idx3-ubyte.gz')
# show_data_set(dataa, 1000)
