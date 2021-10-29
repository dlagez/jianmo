import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
# 选出三个变量与因变量，画出他们之间的散点图。

# 取出数据和标签
# 基于深度学习的方法，BP神经网络，自己定义的网络
data = pd.read_excel("data/Molecular_Descriptor.xlsx", sheet_name=0)
label = pd.read_excel("data/ERα_activity.xlsx", sheet_name=0)

frames = [data, label]
result = pd.concat(frames, axis=1)


# 绘制散点图矩阵, 20个变量太多了，先画5个
data_5 = result.iloc[:, [1, 2, 5, 6, 732]]
sns.pairplot(data_5, height=2.5)
plt.tight_layout()
plt.show()


