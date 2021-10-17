import pandas as pd
data_1972 = pd.read_excel("/content/drive/MyDrive/data/jianmo/Molecular_Descriptor.xlsx", sheet_name=0)
label = pd.read_excel("/content/drive/MyDrive/data/jianmo/ERα_activity.xlsx", sheet_name=0)
data = pd.read_csv("/content/drive/MyDrive/data/jianmo/data_20.csv")

# 取出数据和标签
label_num = label['pIC50'].values.reshape(-1, 1)

data_numpy = data.values
data_num = data_numpy[:, 1:]
data_num = data_num / data_num.max(axis=0)

train_data = data_num[:1500, :].astype(float)
train_label = label_num[:1500, :].astype(float)

test_data = data_num[1500:, :].astype(float)
test_label = label_num[1500:, :].astype(float)

import torch
import torch.nn as nn
# 将数据转换成tensor形式
train_features = torch.tensor(train_data, dtype=torch.float)
train_labels = torch.tensor(train_label, dtype=torch.float)

test_features = torch.tensor(test_data, dtype=torch.float)
test_labels = torch.tensor(test_label, dtype=torch.float)

import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, features):
        super(Net, self).__init__()

        self.linear_relu1 = nn.Linear(features, 128)
        self.linear_relu2 = nn.Linear(128, 256)
        self.linear_relu3 = nn.Linear(256, 256)
        self.linear_relu4 = nn.Linear(256, 256)
        self.linear5 = nn.Linear(256, 1)

    def forward(self, x):
        y_pred = self.linear_relu1(x)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu2(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu3(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu4(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear5(y_pred)
        return y_pred

model = Net(features=data_num.shape[1])
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

losses = []
i = 0
# 训练500轮
for t in range(500):
    y_pred = model(train_features)

    loss = criterion(y_pred, train_labels)
    # print(t, loss.item())
    losses.append(loss.item())

    if torch.isnan(loss):
        break

    # 将模型中各参数的梯度清零。
    # PyTorch的backward()方法计算梯度会默认将本次计算的梯度与缓存中已有的梯度加和。
    # 必须在反向传播前先清零。
    optimizer.zero_grad()

    # 反向传播，计算各参数对于损失loss的梯度
    loss.backward()

    # 根据刚刚反向传播得到的梯度更新模型参数
    optimizer.step()
    print(losses[i])
    i = i + 1

predictions = model(test_features).detach().numpy()

import numpy as np
print(predictions.shape)
print(predictions.reshape(-1))
np.savetxt("pred.txt", predictions.reshape(-1))