# FERNet
基于深度学习的面部表情识别 (Facial-expression Recognition)
## 一、项目背景
数据集[cnn_train.csv](https://download.csdn.net/download/qq_45588019/21981932)包含人类面部表情的图片的label和feature。在这里，面部表情识别相当于一个分类问题，共有7个类别。<br>
其中label包括7种类型表情：<br>
![7-classes](https://gitee.com/zhou-zhou123c/FERNet/raw/master/result/images/7-classes.png)<br>
一共有28709个label，说明包含了28709张表情包。<br>
每一行就是一张表情包4848=2304个像素，相当于4848个灰度值(intensity)(0为黑, 255为白)
## 二、数据预处理
### 1.标签与特征分离
[cnn_feature_label.py](https://github.com/HaoliangZhou/FERNet/blob/master/dataloader/cnn_feature_label.py)<br>
对[原数据](https://download.csdn.net/download/qq_45588019/21981932)进行处理，分离后分别保存为cnn_label.csv和cnn_data.csv.()
### 2.数据可视化
[face_view.py](https://github.com/HaoliangZhou/FERNet/blob/master/dataloader/face_view.py)<br>
对特征进一步处理，也就是将每个数据行的2304个像素值合成每张48*48的表情图，最后做成24000张表情包。
### 3.分割训练集和测试集
[cnn_picture_label.py](https://github.com/HaoliangZhou/FERNet/blob/master/dataloader/cnn_picture_label.py)<br>
__Step1__:划分一下训练集和验证集。一共有28709张图片，我取前24000张图片作为训练集，其他图片作为验证集。新建文件夹cnn_train和cnn_val，将0.jpg到23999.jpg放进文件夹cnn_train，将其他图片放进文件夹cnn_val.<br>
__Step2__:对每张图片标记属于哪一个类别，存放在dataset.csv中，分别在刚刚训练集和测试集执行标记任务。<br>
__Step3__:重写Dataset类，它是Pytorch中图像数据集加载的一个基类，需要重写类来实现加载上面的图像数据集 ([rewrite_dataset.py](https://github.com/HaoliangZhou/FERNet/blob/master/dataloader/rewrite_dataset.py))
## 三、搭建模型
[CNN_face.py](https://github.com/HaoliangZhou/FERNet/blob/master/models/CNN_face.py)<br>
![neural_network](https://gitee.com/zhou-zhou123c/FERNet/raw/master/result/images/neural_network.jpg width="100px")
## 四、训练模型
[train.py](https://github.com/HaoliangZhou/FERNet/blob/master/train.py)<br>
损失函数使用交叉熵，优化器是随机梯度下降SGD，其中weight_decay为正则项系数，每轮训练打印损失值，每5轮训练打印准确率。<br>
源数据放在[CSDN](https://download.csdn.net/download/qq_45588019/21981932)
