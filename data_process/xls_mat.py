import pandas as pd

# 取出数据和标签
# 基于深度学习的方法，BP神经网络，自己定义的网络
data = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=0)
label = pd.read_excel("ERα_activity.xlsx", sheet_name=0)


label_num = label['pIC50'].values.reshape(-1, 1)

data_numpy = data.values
data_num = data_numpy[:, 1:]
print(data_num.shape)

print(label_num.shape)

train_data = data_num[:1500, :]
train_label = label_num[:1500, :]
train_data.astype(float)
print(train_data.dtype)
test_data = data_num[1500:, :]
test_label = label_num[1500:, :]

print("train_data.shape=", train_data.shape, "train_label.shape=", train_label.shape)
print("test_data.shape=", test_data.shape, "test_label.shape=", test_label.shape)
