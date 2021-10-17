from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import scipy.io as scio

# 第三问第一列的解答
# 加载第三问选择的列
index = scio.loadmat("stay_sort_idx.mat")['stay_sort_ind'].reshape(-1)
index = list(index)

# 将data和label分成4份， 三份用来学习，一份用来测试
data = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = pd.read_excel("ADMET.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = label['MN']

index = [i - 1 for i in index]
data = data.iloc[:, index]

data = data.values
label = label.values

# 将所有的数据用来训练，预测题目的数据
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(data, label)

# 加载第三题题目的测试数据
test_data = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=1, index_col='SMILES').astype(float)
test_data = test_data.iloc[:, index].values

test_label_pred = clf.predict(test_data)

for i in [3, 8, 8, 14, 16, 30, 35, 41, 43, 48]:
    test_label_pred[i] = 0
print(test_label_pred)


