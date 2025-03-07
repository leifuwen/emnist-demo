"""
模型训练 letter_train.py
"""
import random
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
from src.utils.load_emnist import *

'''
python 3.8
tensorflow 2.0.0b0
'''

"""
配置项
"""
config = {
    # 训练好的模型存放路径
    'check_path': "./ckpt/cp-{epoch:04d}.ckpt",
    # 类别数
    'class_num': 27,
    # 数据集位置
    'data_sets': [
        './../../data_set_emnist_letters/emnist-letters-train-labels-idx1-ubyte.gz',
        './../../data_set_emnist_letters/emnist-letters-train-images-idx3-ubyte.gz',
        './../../data_set_emnist_letters/emnist-letters-test-labels-idx1-ubyte.gz',
        './../../data_set_emnist_letters/emnist-letters-test-images-idx3-ubyte.gz']
}


class CNN(object):
    def __init__(self):
        self.model = self.model()

    @staticmethod
    def model():
        """
        LetNet模型，测试准确度在94左右
        :return: model
        """
        model = models.Sequential()
        # 第1层卷积：卷积核大小为3*3，32个，28*28为待训练图片的大小，激活函数选用relu
        #   input_shape=(28, 28, 1) 输入层的维度
        #       此处三个数分别代表图片的长、宽、颜色深度（图片已经转化为灰度图像，所以颜色深度是1）
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        # 池化层，减少参数(降维)和计算量，尽可能无损地提取主要特征，提高模型的适应能力
        #   参数是沿（垂直，水平）方向缩小比例的因数。（2, 2）会把输入张量的两个维度都缩小一半
        model.add(layers.MaxPooling2D((2, 2)))
        # 第2层卷积：卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # 池化层
        model.add(layers.MaxPooling2D((2, 2)))
        # 第3层卷积：卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        """
        模型的后半部分，是定义输出张量的。
        layers.Flatten会将三维的张量转为一维的向量。
        展开前张量的维度是(3, 3, 64) ，转为一维(576)的向量后，
        紧接着使用layers.Dense层，构造了2层全连接层，
        逐步地将一维向量的位数从28*28=576变为64，再变为27。
        """

        # 将上一层的输出降到一维，即“扁平化”。（从卷积层到全连接层的过渡）
        model.add(layers.Flatten())
        # 后半部分相当于是构建了一个隐藏层为64，输入层为28*28=576，输出层为27的普通的神经网络
        model.add(layers.Dense(64, activation='relu'))
        # 输出层，27代表26个英文字母+1
        model.add(layers.Dense(config['class_num'], activation=tf.nn.softmax, name='predictions'))
        # 打印这个模型的总体参数
        model.summary()
        return model


class DataSourceEMnist(object):
    """
    获取数据集
    """

    def __init__(self):
        files = config['data_sets']

        # 读取数据集
        # train_labels = read_idx1(files[0])
        # train_images, train_images_num = read_idx3(files[1])
        test_labels = read_idx1(files[2])
        test_images, test_images_num = read_idx3(files[3])
        train_images = np.load('./../../data_set_self_letters/train_images_s.npy')
        train_labels = np.load('./../../data_set_self_letters/train_labels_s.npy')
        my_train_images, my_train_images_num = read_img("./../../data_set_self_letters/img")
        my_train_lables = read_img_lable("./../../data_set_self_letters/label.txt")
        mycsv_train_images, mycsv_train_images_num = read_img("./../../data_set_self_letters/imgcsv")
        mycsv_train_lables = read_csv_lable("./../../data_set_self_letters/english.csv")

        test_images = test_images[test_labels <= 4, :, :]
        test_labels = test_labels[test_labels <= 4]

        train_images_l = []
        # train_labels_l = []
        # # 根据标签筛选数据
        # for i, val in enumerate(train_labels):
        #     if val <= 4:
        #         print(i, train_labels[i])
        #         train_images_l.append(train_images[i])
        #         train_labels_l.append(val)
        # train_images_s = np.array(train_images_l)
        # train_labels_s = np.array(train_labels_l)
        # np.save("./../../data_set_self_letters/train_images_s.npy", train_images_s)
        # np.save("./../../data_set_self_letters/train_labels_s.npy", train_labels_s)

        # 将Emnist数据集图像水平翻转为正常图像
        for x in test_images:
            img = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.flip(img, 1)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            train_images_l.append(img)
        test_images = np.array(train_images_l)
        # np.save('./../../data_set_self_letters/train_images_s.npy', train_images_s)

        # reshape()中的第一个参数传入训练集（测试集）的大小，下同。
        train_images = train_images.reshape((len(train_images), 28, 28, 1))
        test_images = test_images.reshape((len(test_images), 28, 28, 1))
        my_train_images = my_train_images.reshape((my_train_images_num, 28, 28, 1))
        mycsv_train_images = mycsv_train_images.reshape((mycsv_train_images_num, 28, 28, 1))

        # 像素值映射到 0 - 1 之间
        train_images, test_images, my_train_images, mycsv_train_images = (train_images / 255.0,
                                                                          test_images / 255.0,
                                                                          my_train_images / 255.0,
                                                                          mycsv_train_images / 255.0)
        # 以相同的随机种子打乱训练图片和标签
        random.seed(7)
        random.shuffle(my_train_images)
        random.seed(7)
        random.shuffle(my_train_lables)
        random.seed(7)
        random.shuffle(mycsv_train_images)
        random.seed(7)
        random.shuffle(mycsv_train_lables)

        # 合并
        images = np.concatenate((train_images, mycsv_train_images, my_train_images), axis=0)
        labels = np.concatenate((train_labels, mycsv_train_lables, my_train_lables), axis=0)

        self.train_images, self.train_labels = images, labels
        self.test_images, self.test_labels = test_images, test_labels


class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSourceEMnist()

    def train(self):
        check_path = config['check_path']
        # 保存模型到“检查点”，在后面完成训练模型时作为一个回调使用
        #   只能是 `keras.callbacks.Callback` 的实例
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(
            check_path, save_weights_only=True, verbose=1, steps_per_execution=5)
        # 编译这个模型
        #   optimizer: 优化器名或者优化器实例
        #   loss: 损失函数名
        #   metrics: 在训练和测试期间的模型评估标准
        self.cnn.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        # 开始训练。
        #   前两个参数传入训练图片和标记的标签
        #   epochs=5 训练五轮
        #   callback 回调，表示训练完毕之后做什么。只能是 `keras.callbacks.Callback` 的实例
        self.cnn.model.fit(self.data.train_images, self.data.train_labels,
                           epochs=5, callbacks=[save_model_cb])
        # 使用测试集对模型进行测试和评估
        test_loss, test_acc = self.cnn.model.evaluate(
            self.data.test_images, self.data.test_labels)
        print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(self.data.test_labels)))


if __name__ == "__main__":
    app = Train()
    app.train()
