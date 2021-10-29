import pandas as pd
import scipy.io as scio

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.svm import NuSVR

# 比较mlp和decissionTree的效果
index = scio.loadmat("data/res_idx_20.mat")['stay'].reshape(-1)
index = list(index)

# 加载示例函数，我看看它张什么样
data = pd.read_excel("data/Molecular_Descriptor.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = pd.read_excel("data/ERα_activity.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = label['pIC50']

index = [i - 1 for i in index]
data = data.iloc[:, index]

data = data.values
label = label.values

tree = DecisionTreeRegressor()
mlp = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(100, 100), tol=1e-2, max_iter=20000, random_state=0),
)
regr = make_pipeline(StandardScaler(), NuSVR(C=1.0, nu=0.1))


tree.fit(data, label)
mlp.fit(data, label)
regr.fit(data, label)


# 将Decision Tree 和 MLP 的局部依赖图分开画在两个图上面。
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("Decision Tree")
tree_disp = PartialDependenceDisplay.from_estimator(tree, data, [1, 2, 3, 6], ax=ax)


fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("Multi-layer Perceptron")
mlp_disp = PartialDependenceDisplay.from_estimator(
    mlp, data, [1, 2, 3, 6], ax=ax, line_kw={"color": "red"}
)

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("Support Vector Machines")
regr_disp = PartialDependenceDisplay.from_estimator(regr, data, [1, 2, 3, 6], ax=ax)


# plt.show()


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 6))
tree_disp.plot(ax=[ax1, ax2, ax3, ax4], line_kw={"label": "Decision Tree"})
mlp_disp.plot(
    ax=[ax1, ax2, ax3, ax4], line_kw={"label": "Multi-layer Perceptron", "color": "red"}
)
regr_disp.plot(ax=[ax1, ax2, ax3, ax4], line_kw={"label": "Support Vector Machines", "color": "green"})
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax1.set_xlabel('AMR')
ax2.set_xlabel('ATSc3')
ax3.set_xlabel('nsssNHp')
ax4.set_xlabel('SddssSe')
plt.show()
