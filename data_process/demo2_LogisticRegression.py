from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

print('X.shape: ', X.shape, "y.shape: ", y.shape)
print(X.dtype), print(y.dtype)
data = pd.read_csv("data_20.csv", index_col="SMILES").astype(float)
data = data.values

label = pd.read_excel("ERα_activity.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = label['pIC50'].values

print('data.shape: ', data.shape, " label.shape: ", label.shape)
print(label.dtype)
clf = LogisticRegression(random_state=0).fit(data, label)

