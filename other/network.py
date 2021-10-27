import scipy.io as scio
import pandas as pd
from sklearn import preprocessing

index = scio.loadmat("res_idx_20.mat")['stay']
index = list(index.reshape(-1))
index_new = [i - 1 for i in index]

data = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = pd.read_excel("ERα_activity.xlsx", sheet_name=0, index_col='SMILES').astype(float)


data_20 = data.iloc[:, index_new]
print(data_20)
data_numpy = data_20.values

data_4 = data.iloc[:, [4]]

data_20_array = data_20.values


index_XGBoost = [22, 40, 24, 38, 21, 0, 1, 384, 658, 584, 290, 39, 726, 23, 586, 102, 528, 391, 503, 343]
data_20 = data.iloc[:, index_XGBoost]

data_20.to_csv("data_20.csv")

min_max_scaler = preprocessing.MinMaxScaler()

data_min_max = min_max_scaler.fit_transform(data_20)
print(data_min_max.shape)
data_min_max.to_csv("data_20_min_max.csv")