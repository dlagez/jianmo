import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("data_20.csv", index_col="SMILES").astype(float)

label = pd.read_excel("ERα_activity.xlsx", sheet_name=0, index_col='SMILES').astype(float)
cols = data.columns.values.tolist()
print(len(cols))  # 20


# 绘制散点图矩阵
sns.pairplot(data, size=2.5)
plt.tight_layout()
plt.show()


# 绘制热度图
cm = np.corrcoef(data.values.T)
print(cm.shape)

hm = sns.heatmap(cm,
                 cbar=True,

                 square=True,
                 xticklabels=cols,
                 yticklabels=cols

                 )



plt.show()

