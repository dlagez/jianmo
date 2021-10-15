import pandas as pd
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.decomposition import SparsePCA


# 取出数据和标签
# 基于深度学习的方法，BP神经网络，自己定义的网络
data = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=0)
label = pd.read_excel("ERα_activity.xlsx", sheet_name=0)


label_num = label['pIC50'].values.reshape(-1, 1)

data_numpy = data.values
data_num = data_numpy[:, 1:]

print(data_num.shape)
print(label_num.shape)

X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
print(X.shape)

transformer = SparsePCA(n_components=5, random_state=0)
transformer.fit(data_num)

X_transformed = transformer.transform(data_num)
X_transformed.shape

a = transformer.components_
print(a.shape)