import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio

index = scio.loadmat("stay_sort_idx.mat")['stay_sort_ind'].reshape(-1)
index = list(index)
data = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = pd.read_excel("ERα_activity.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = label['pIC50'].values
label = list(label)

index = [i - 1 for i in index]
data = data.iloc[:, index].values

# 绘制散点图矩阵
plt.figure()
plt.rcParams['figure.figsize'] = (50.0, 50.0)  # 设置figure_size尺寸
for i in range(1, 395):
    plt.subplot(20, 20, i)
    plt.scatter(list(data[:, i-1]), label)
plt.show()

list(data[:, 1])

