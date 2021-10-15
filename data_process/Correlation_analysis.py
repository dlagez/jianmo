import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as status
import seaborn as sns


data = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = pd.read_excel("ERα_activity.xlsx", sheet_name=0, index_col='SMILES').astype(float)
data = data.iloc[:100, :100]

label_num = label['pIC50'].values.reshape(-1, 1)

data_numpy = data.values
data_num = data_numpy[:, 1:]

X = data['XLogP']
Y = data['Zagreb']
result = np.corrcoef(X, Y)

result2 = data.corr()

# 代码执行不出来
sns.pairplot(data)
print(data)

figure, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), square=True, axax=ax)
