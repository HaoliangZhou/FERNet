# ###二、数据可视化,将每个数据行的2304个像素值合成每张48*48的表情图。
import cv2
import numpy as np

# 放图片的路径
path = '../images'
# 读取像素数据
data = np.loadtxt('../datasets/cnn_data.csv')

# 按行取数据并写图
for i in range(data.shape[0]):
    face_array = data[i, :].reshape((48, 48))  # reshape
    cv2.imwrite(path + '//' + '{}.jpg'.format(i), face_array)  # 写图片
