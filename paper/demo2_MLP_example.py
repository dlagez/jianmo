import pandas as pd
import scipy.io as scio
from sklearn import neural_network
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import cross_val_predict
import numpy as np
import matplotlib.pyplot as pl


# 看到网上有代码，直接用了，它是用来比较多层感知机不同的激活函数拟合出来的效果
index = scio.loadmat("data/res_idx_20.mat")['stay'].reshape(-1)
index = list(index)

# 将data和label分成4份， 三份用来学习，一份用来测试
data = pd.read_excel("data/Molecular_Descriptor.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = pd.read_excel("data/ERα_activity.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = label['pIC50']


index = [i - 1 for i in index]
data = data.iloc[:, index]

data = data.values
label = label.values

n_fig=0
for name, nn_unit in [
        ('MLP using ReLU', neural_network.MLPRegressor(activation='relu', solver='adam')), ('MLP using Logistic Neurons', neural_network.MLPRegressor(activation='logistic')), ('MLP using TanH Neurons', neural_network.MLPRegressor(activation='tanh',solver='adam'))
        ]:
    regressormodel=nn_unit.fit(data, label)
    # Y predicted values
    yp =nn_unit.predict(data)
    rmse =np.sqrt(mean_squared_error(label, yp))  # 均匀的误差回归损失。 np.sqrt 返回数组的非负平方根
    #Calculation 10-Fold CV
    yp_cv = cross_val_predict(regressormodel, data, label, cv=10)  # 为每个输入数据点生成交叉验证的估计
    rmsecv=np.sqrt(mean_squared_error(label,yp_cv))  # 均匀的误差回归损失
    print('Method: %s' %name)
    print('RMSE on the data: %.4f' %rmse)
    print('RMSE on 10-fold CV: %.4f' %rmsecv)
    n_fig=n_fig+1
    pl.figure(n_fig)
    pl.plot(yp, label,'ro', label='predict')
    pl.plot(yp_cv, label,'bo', alpha=0.25, label='10-folds CV')
    pl.xlabel('predicted')
    pl.title('Method: %s' %name)
    pl.ylabel('real')
    pl.legend()
    pl.grid(True)
    pl.show()