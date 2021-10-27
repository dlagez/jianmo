import matplotlib.pyplot as plt
import pandas as pd

# 选出三个变量与因变量，画出他们之间的散点图。

# 取出数据和标签
# 基于深度学习的方法，BP神经网络，自己定义的网络
data = pd.read_excel("data/Molecular_Descriptor.xlsx", sheet_name=0)
label = pd.read_excel("data/ERα_activity.xlsx", sheet_name=0)

frames = [data, label]
result = pd.concat(frames, axis=1)

X = result['apol']  # SsLi nB
Y = result['pIC50']

# apol ATSc3 ATSm2 BCUTp-1h SCH-7
result.plot.scatter(x='apol', y='pIC50')
plt.show()

