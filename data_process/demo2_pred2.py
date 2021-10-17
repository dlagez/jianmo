import numpy
import scipy.io as scio
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

pred = scio.loadmat('data.mat')['data'].reshape(-1)
label = pd.read_excel("ERα_activity.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = label['pIC50'].values

pred = list(pred)
label = list(label)

label = label[1500:]

index = list(range(474))

plt.rcParams['figure.figsize'] = (25.0, 8.0) # 设置figure_size尺寸
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_title('真实值和预测值的对比')
ax.set_xlabel('样本数量')
ax.set_ylabel('预测与真实值')
ax.plot(index, pred, color='green', label='预测')
ax.plot(index, label, color='red', label='真实值')
plt.legend(loc='best')
plt.show()


