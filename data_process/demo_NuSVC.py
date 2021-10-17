import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC

data = pd.read_excel('data_num_clear.xlsx', sheet_name=0, index_col='SMILES').astype(float)
label = pd.read_excel("ADMET.xlsx", sheet_name=0, index_col='SMILES').astype(float)
data = data.values
label = label['MN'].values

train_data = data[:1500, :]
test_data = data[1500:, :]

train_label = label[:1500]
test_label = label[1500:]

train_data = list(train_data)
test_data = list(test_data)

train_label = list(train_label)
test_label = list(test_label)

clf = make_pipeline(StandardScaler(), NuSVC(nu=0.1))
clf.fit(train_data, train_label)
pred = clf.predict(test_data)
print(pred)

from sklearn.metrics import accuracy_score
accuracy_score(test_label, pred)

1

