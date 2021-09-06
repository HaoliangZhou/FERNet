# ###一、将原始数据的label和feature(像素)数据分离
import pandas as pd

# 源数据路径
path = '../datasets/originalData/cnn_train.csv'
# 读取数据
df = pd.read_csv(path)
# 提取feature(像素)数据 和 label数据
df_x = df[['feature']]
df_y = df[['label']]
# 将feature和label数据分别写入两个数据集
df_x.to_csv('../datasets/cnn_data.csv', index=False, header=False)
df_y.to_csv('../datasets/cnn_label.csv', index=False, header=False)
