import pandas as pd
# 韦继业组的混淆举证
data = pd.read_excel('wei.xlsx', sheet_name=0)

caco_2 = data['Caco-2']
caco_2_pred = data['Caco-2-pre']

CYP3A4 = data['CYP3A4']
CYP3A4_pre = data['CYP3A4-pre']

hERG = data['hERG']
hERG_pre = data['hERG-pre']

HOB = data['HOB']
HOB_pre = data['HOB-pre']

MN = data['MN']
MN_pre =data['MN-pre']



from sklearn.metrics import confusion_matrix
confusion_matrix(caco_2, caco_2_pred)
# array([[421,  17],
#        [ 41, 115]])

confusion_matrix(CYP3A4, CYP3A4_pre)
# array([[ 49,  31],
#        [ 13, 501]])

confusion_matrix(hERG, hERG_pre)
# array([[107,  61],
#        [ 52, 374]])

confusion_matrix(HOB, HOB_pre)
# array([[382,  83],
#        [ 67,  62]])

confusion_matrix(MN, MN_pre)
# array([[136,  29],
#        [104, 325]])