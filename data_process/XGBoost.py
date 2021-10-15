import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston

boston = load_boston()
df = pd.read_csv("diabetes.csv")
print(df.shape)
data = df.iloc[:, :8]
target = df.iloc[:, -1]

train_x, test_x, train_y, test_y = train_test_split(data,target,test_size=0.2,random_state=7)

iris = load_iris()

print(boston['data'].shape)
print(iris['data'].shape)