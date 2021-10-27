import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 画出每个特征的权重图
result = pd.read_csv('result.csv')
index = result['index'].values
index = [i - 1 for i in index]

# 读取所有特征
data = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=0, index_col='SMILES').astype(float)
col = list(data.columns.values)
select = [col[i] for i in index]
select = sorted(select, reverse=True)
# 对应的权重
importance = result['importance'].values
importance = sorted(importance, reverse=True)

index_col = [i for i in range(1, 375)]

fig = plt.figure()
plt.errorbar(index_col, importance)
plt.legend(loc='lower right')
plt.show()