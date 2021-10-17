import numpy
import scipy.io as scio
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
import random
# value = []
loss = [1.28, 1.23, 1.17, 1.09, 0.98, 0.91, 0.86, 0.81, 0.77, 0.65,      0.63, 0.61, 0.59, 0.57, 0.58, 0.56, 0.59, 0.54, 0.55, 0.54]
# for i in range(50):
#     value.append(random.choice([0.09, 0.08, 0.07]))
# loss = loss + value

index = list(range(20))

plt.rcParams['figure.figsize'] = (10, 10) # 设置figure_size尺寸

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.xticks(range(21))
ax.set_title('特征数与损失')
ax.set_xlabel('特征数')
ax.set_ylabel('损失值')
ax.plot(index, loss, color='green')
plt.show()




