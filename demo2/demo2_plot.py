import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("data/data_20.csv", index_col="SMILES").astype(float)

label = pd.read_excel("data/ERα_activity.xlsx", sheet_name=0, index_col='SMILES').astype(float)
cols = data.columns.values.tolist()
print(len(cols))  # 20

# example
penguins = sns.load_dataset("penguins")

# 绘制散点图矩阵, 20个变量太多了，先画5个
data_5 = data.iloc[:, [1, 2, 5, 6, 7]]  # 最后决定这个图。
sns.pairplot(data_5, height=2.5)
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

