import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 海洋组的散点图和热度图
# data = pd.read_csv("data_20.csv", index_col="SMILES").astype(float)

data = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=0, index_col='SMILES').astype(float)

label = pd.read_excel("ERα_activity.xlsx", sheet_name=0, index_col='SMILES').astype(float)
# cols = data.columns.values.tolist()
# print(len(cols))  # 20
cols = ['AMR', 'apol', 'nC', 'nBonds', 'nBonds2', 'SP-3', 'VP-1', 'VP-2', 'VP-3', 'minsssN', 'maxsssCH', 'maxsssN',	'hmin',	'LipoaffinityIndex', 'ETA_Alpha', 'MLogP', 'MDEC-23', 'VAdjMat', 'WTPT-1', 'Zagreb']


sns.pairplot(data, size=2.5)

plt.tight_layout()
plt.show()


cm = np.corrcoef(data[cols].values.T)
print(cm.shape)

hm = sns.heatmap(cm,
                 cbar=True,

                 square=True,
                 xticklabels=cols,
                 yticklabels=cols

                 )

plt.show()

