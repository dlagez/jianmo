import matplotlib.pyplot as plt
import pandas as pd

# 取出数据和标签
# 基于深度学习的方法，BP神经网络，自己定义的网络
data = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=0)
label = pd.read_excel("ERα_activity.xlsx", sheet_name=0)

frames = [data, label]
result = pd.concat(frames, axis=1)

X = result['SsLi']
Y = result['pIC50']

result.plot.scatter(x='SsLi', y='pIC50')
plt.show()

