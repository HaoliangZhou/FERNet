# ###三、表情图片和类别标注,
# 1.取前24000张图片作为训练集放入cnn_train，其他图片作为验证集放入cnn_val
# 2.对每张图片标记属于哪一个类别，存放在dataset.csv中，分别在刚刚训练集和测试集执行标记任务。

# #因为cpu训练太慢，我只取前2000张做训练，400张做测试！！，手动删除两个文件夹重dataset.csv的多余行数据
import os
import pandas as pd


def data_label(path):
    # 读取label文件
    df_label = pd.read_csv('../datasets/cnn_label.csv', header=None)
    # 查看该文件夹下所有文件
    files_dir = os.listdir(path)
    # 存放文件名和标签的列表
    path_list = []
    label_list = []
    # 遍历所有文件，取出文件名和对应的标签分别放入path_list和label_list列表
    for file_dir in files_dir:
        if os.path.splitext(file_dir)[1] == '.jpg':
            path_list.append(file_dir)
            index = int(os.path.splitext(file_dir)[0])
            label_list.append(df_label.iat[index, 0])
    # 将两个列表写进dataset.csv文件
    path_s = pd.Series(path_list)
    label_s = pd.Series(label_list)
    df = pd.DataFrame()
    df['path'] = path_s
    df['label'] = label_s
    df.to_csv(path + '\\dataset.csv', index=False, header=False)


def main():
    # 指定文件夹路径
    train_path = '../datasets/cnn_train'
    val_path = '../datasets/cnn_val'
    data_label(train_path)
    data_label(val_path)


if __name__ == '__main__':
    main()











