import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# 根据房价预测的模型来解决S

# 解决第二问问题的代码，生成了预测的数据
df = pd.read_csv("data_20.csv", index_col="SMILES").astype(float)
label = pd.read_excel("ERα_activity.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = label['pIC50']
cols = df.columns.values.tolist()

# 读取题目需要的预测的数据
data_test = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=1, index_col='SMILES').astype(float)
label_test = pd.read_excel("ERα_activity.xlsx", sheet_name=1, index_col='SMILES').astype(float)

data_test = data_test[cols]

# 显示各不同特征两两之间的关系
sns.set(style='whitegrid', context='notebook')

sns.pairplot(df)
plt.tight_layout()
plt.show()

# 画热力图
print(df.values.T.shape)

cm = np.corrcoef(df.values.T)
hm = sns.heatmap(cm,
                 cbar=True,
                 square=True,
                 fmt='.2f',
                 yticklabels=cols,
                 xticklabels=cols)
plt.show()


# 这里对应问题二，多元线性回归模型=================================

from sklearn.model_selection import train_test_split

X = df.values
y = label.values
print(X.shape, y.shape)
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=0)

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)


# 第二题题解：
pred = forest.predict(data_test)

data_frame = {'pred': pred}
frame = pd.DataFrame(data_frame)




print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# 画图
plt.scatter(y_train_pred,
            y_train_pred - y_train,
            c='black',
            marker='o',
            s=35,
            alpha=0.5,
            label='Training data')
plt.scatter(y_test_pred,
            y_test_pred - y_test,
            c='lightgreen',
            marker='s',
            s=35,
            alpha=0.7,
            label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.tight_layout()

plt.show()

# 模型评估
data = pd.read_csv("data_20.csv", index_col="SMILES").astype(float)
label = pd.read_excel("ERα_activity.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = label['pIC50']

train_data = data.iloc[:1500, :]
test_data = data.iloc[1500:, :]

train_label = label.iloc[:1500]
test_label = label.iloc[1500:]


from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)
forest.fit(train_data, train_label)
pred = forest.predict(test_data)
from sklearn.metrics import r2_score
r2_score(test_label, pred, multioutput='variance_weighted')





import matplotlib.pyplot as plt
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(test_label, pred)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.title('NuSVC MN Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
