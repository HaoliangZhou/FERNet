# ###四、重写类来实现加载上面的图像数据集。
import bisect
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class FaceDataset(data.Dataset):
    # 初始化
    def __init__(self, root):
        super(FaceDataset, self).__init__()
        self.root = root
        df_path = pd.read_csv(root + '\\dataset.csv', header=None, usecols=[0])
        df_label = pd.read_csv(root + '\\dataset.csv', header=None, usecols=[1])
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    # 读取某幅图片，item为索引号
    def __getitem__(self, item):
        # 图像数据用于训练，需为tensor类型，label用numpy或list均可
        face = cv2.imread(self.root + '\\' + self.path[item])
        # 读取单通道灰度图
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # 直方图均衡化
        face_hist = cv2.equalizeHist(face_gray)

        """
        像素值标准化
        读出的数据是48X48的，而后续卷积神经网络中nn.Conv2d() 
        API所接受的数据格式是(batch_size, channel, width, height)，
        本次图片通道为1，因此我们要将48X48 reshape为1X48X48。
        """
        face_normalized = face_hist.reshape(1, 48, 48) / 255.0
        face_tensor = torch.from_numpy(face_normalized)
        face_tensor = face_tensor.type('torch.FloatTensor')
        label = self.label[item]
        return face_tensor, label

    # 获取数据集样本个数
    def __len__(self):
        return self.path.shape[0]








