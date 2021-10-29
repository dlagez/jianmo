import pandas as pd
import scipy.io as scio
from sklearn.neural_network import MLPRegressor

# 使用多层感知机回归。
# 预测第一问第一列的网络
# 加载第三问选择的列
index = scio.loadmat("data/res_idx_20.mat")['stay'].reshape(-1)
index = list(index)

# 将data和label分成4份， 三份用来学习，一份用来测试
data = pd.read_excel("data/Molecular_Descriptor.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = pd.read_excel("data/ERα_activity.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = label['pIC50']


index = [i - 1 for i in index]
data = data.iloc[:, index]


train_data = data.iloc[:1500, :]
train_label = label.iloc[:1500]

test_data = data.iloc[1500:, :]
test_label = label.iloc[1500:]


# 将数据编程array
train_data = train_data.values
train_label = train_label.values
test_data = test_data.values
test_label = test_label.values

regr = MLPRegressor(random_state=1, max_iter=5000).fit(train_data, train_label)
result_label = regr.predict(test_data)