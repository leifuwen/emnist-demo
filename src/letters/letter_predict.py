"""
调用模型进行单个字符识别 letter_predict.py
"""
import time
import cv2
import tensorflow as tf
# import numpy as np
import os as os
from matplotlib import pyplot as plt
from tinydb import TinyDB, Query
from PIL import Image, ImageEnhance
from src.letters.letter_train import CNN
from src.utils.visualize import *

'''
python 3.8
tensorflow 2.0.0b0
pillow(PIL) 4.3.0
'''


class Predict(object):

    def __init__(self):
        latest = tf.train.latest_checkpoint('./ckpt')
        self.cnn = CNN()
        # 恢复网络权重
        self.cnn.model.load_weights(latest).expect_partial()

    def predict(self, image_path):
        """
        预测单个字符
        :param image_path:图片路径
        :return: 字符的ascll码
        """
        # 读取图片并转换为灰度模式，即将RGB通道合为一个通道，对这个单通道的数据进行操作
        img_c = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 阈值化
        ret, img_c = cv2.threshold(img_c, 140, 255, cv2.THRESH_BINARY)
        # 将图片放大
        img_c = cv2.resize(img_c, (28, 28))[5:23, 5:23]
        img_c = cv2.resize(img_c, (28, 28))
        # reshape()改变数组的结构，并且原始数据不发生变化
        # 将这个灰度图片作为一个数组，转换为28大组、28小组、每小组元素个数为1的数组
        # / 255 表示每个数都除以255，将每个数都压缩到0-1之间
        img_p = np.reshape(img_c, (28, 28, 1)) / 255

        # 数组升一维，是为了符合模型的维度
        x = np.array([img_p])
        # 开始根据模型预测
        y = self.cnn.model.predict(x)

        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得最大值的下标，即代表的数字
        # print(np.argmax(y[0]))
        # print(time.localtime())
        # 可视化的方式展示图片
        # show_pred_result(image_path, get_mapping(np.argmax(y[0]), with_type="letters"))

        return get_mapping(np.argmax(y[0]), with_type="letters")


def prepare_image(img: Image) -> Image:
    """
    将待预测的图片进行预处理：
        水平翻转、逆时针旋转90度并统一缩放到28*28px。

    这是因为，EMNIST中的数据集中参与训练的图片，
    是经过水平反转并顺时针旋转90度的，
    所以在预测的时候也要对应地变换一下。
    （所以EMNIST为什么要这么处理图片呢？）
    :param img:
    :return: Image
    """
    img = img.crop((60, 60, 255, 255))
    # plt.imshow(img)
    # plt.show()
    return img \
        .transpose(Image.Transpose.FLIP_LEFT_RIGHT) \
        .transpose(Image.Transpose.ROTATE_90) \
        .resize((28, 28), Image.Resampling.LANCZOS)


def answer_correct():
    """
    开始预测，并与数据库对比答案
    :return: 字典列表
    """
    image_dir = '../../english_images/'
    # cwd = os.getcwd()
    files = os.listdir(image_dir)

    # 遍历批量导入预测文件夹内的所有图片
    app = Predict()
    # for file in files:
    #     result = app.predict(image_dir + file)
    # 连接数据库
    db = TinyDB('database/letters.json')
    # 读取数据表
    tb = db.table('letters')
    answer_list = []
    for image in files:
        # 预测
        result = app.predict(image_dir + image)
        num = int(image.split('.')[0])

        # 获取正确答案
        answer = tb.get(doc_id=num)['answer']
        istrue = answer == result
        answer_json = {"id": num, "answer": answer, "my_answer": result, "is_True": istrue}
        answer_list.append(answer_json)
    db.close()
    return answer_list
