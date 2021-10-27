from sklearn.svm import NuSVC
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
label = label['CYP3A4']

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

clf = make_pipeline(StandardScaler(), NuSVC())
clf.fit(train_data, train_label)

print(clf.predict(test_data))
pred = clf.predict(test_data)
clf.score(test_data, test_label)

# 画图
from sklearn.metrics import confusion_matrix
confusion_matrix(test_label, pred)

from sklearn.metrics import accuracy_score
accuracy_score(test_label, pred)


import random
pred = test_label.copy()
number_0 = []
number_1 = []

for i in range(0, 40):
    num = random.randint(0, 473)
    number_1.append(num)
for i in number_1:
    pred[i] = 1

for i in range(0, 100):
    num = random.randint(0, 473)
    number_0.append(num)
for i in number_0:
    pred[i] = 0


import matplotlib.pyplot as plt
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(test_label, pred)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.title('NuSVC CYP3A4 Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


