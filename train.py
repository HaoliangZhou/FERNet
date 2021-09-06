# 定义训练轮
import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
from torch import optim

from models import CNN_face
from dataloader import rewrite_dataset


def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay, print_cost=True, isPlot=True):
    # 加载数据集并分割batch
    train_loader = data.DataLoader(train_dataset, batch_size)
    # 构建模型
    model = CNN_face.FaceCNN()
    # 损失函数和优化器
    compute_loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # 学习率衰减
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    for epoch in range(epochs):
        loss = 0
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()

        # 打印损失值
        if print_cost:
            print('epoch{}: train_loss:'.format(epoch + 1), loss.item())

        # 评估模型准确率
        if epoch % 10 == 9:
            model.eval()
            acc_train = validate(model, train_dataset, batch_size)
            acc_val = validate(model, val_dataset, batch_size)
            print('acc_train: %.1f %%' % (acc_train * 100))
            print('acc_val: %.1f %%' % (acc_val * 100))

    return model


# 验证模型在验证集上的正确率
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    result, total = 0.0, 0
    for images, labels in val_loader:
        pred = model.forward(images)
        pred = np.argmax(pred.data.numpy(), axis=1)
        labels = labels.data.numpy()
        result += np.sum((pred == labels))
        total += len(images)
    acc = result / total
    return acc


def main():
    train_dataset = rewrite_dataset.FaceDataset(root=r'D:\01 Desktop\JUST_YAN\DeepLearning\Facial-expression_Reg\datasets\cnn_train')
    val_dataset = rewrite_dataset.FaceDataset(root=r'D:\01 Desktop\JUST_YAN\DeepLearning\Facial-expression_Reg\datasets\cnn_val')
    model = train(train_dataset, val_dataset, batch_size=128, epochs=100, learning_rate=0.01,
                  wt_decay=0, print_cost=True, isPlot=True)
    torch.save(model, 'model_net.pkl')  # 保存模型


if __name__ == '__main__':
    main()