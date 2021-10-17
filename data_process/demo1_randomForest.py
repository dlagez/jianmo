from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio

index = scio.loadmat("stay_sort_idx.mat")['stay_sort_ind'].reshape(-1)
index = list(index)

# 将data和label分成4份， 三份用来学习，一份用来测试
data = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = pd.read_excel("ADMET.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = label['Caco-2']

index = [i - 1 for i in index]
data = data.iloc[:, index]

data = data.values
label = label.values

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(data, label)

print(regr.feature_importances_)
importance = regr.feature_importances_

importance = list(importance)
importance.sort(reverse=True)

index = list(range(394))

fig = plt.figure()
plt.errorbar(index, importance)
plt.legend(loc='lower right')
plt.show()


