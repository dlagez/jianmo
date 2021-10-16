from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import scipy.io as scio

# 预测第一问第一列的网络
# 加载第三问选择的列
index = scio.loadmat("stay_sort_idx.mat")['stay_sort_ind'].reshape(-1)
index = list(index)

X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

# 将data和label分成4份， 三份用来学习，一份用来测试
data = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = pd.read_excel("ADMET.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = label['Caco-2']

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

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train_data, train_label)

print(clf.predict(test_data))
clf.score(test_data, test_label)


# 将所有的数据用来训练，预测题目的数据
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(data, label)
print(clf)




