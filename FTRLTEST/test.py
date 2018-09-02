from sklearn.datasets import load_svmlight_file
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np

# tt = np.zeros(12)
# print(tt.shape)
# tt.fill(1)
# aa = np.zeros(12)
# aa.fill(2)
# print(np.multiply(aa,tt))
x_train,y_train = load_svmlight_file("./train_data_process")
print(type(x_train))
mat = x_train.todense()
df1 = pd.DataFrame(mat)
print(type(df1))
# # x_test,y_test = load_svmlight_file("./data/part-00001")
# # # x_train_df = pd.DataFrame(x_train.todense())
# # # # print(x_train_df)
# # # x_train_df.to_csv("x_train_df.csv")
# # lrm = LogisticRegression()
# # lrm.fit(x_train,y_train)
# # y_pred = lrm.predict(x_test)
# #
# # fpr,tpr,thresholds=metrics.roc_curve(y_test,y_pred,pos_label=2)
# # auc = metrics.auc(fpr,tpr)
# # print(auc)
# #y_label = []
# positive = 0
# negtive = 0
# lines = []
# with open("./data/train_data_new",'r') as file:
#     line_num = 0
#     for l in file.readlines():
#         new_line = []
#         line_num+=1
#         arr = l.split()
#         if(len(arr)<1): continue
#         if(int(arr[0])>0):
#             positive+=1
#         else:
#             negtive+=1
#         pre_index = 0
#         x_arr = []
#         for i in range(1,len(arr)):
#             index = arr[i].split(":")[0]
#             if(int(index)<=pre_index):
#                 continue
#             else:
#                 x_arr.append(arr[i])
#                 pre_index=int(index)
#         new_line = str(arr[0])+" "+" ".join(x_arr)
#         lines.append(new_line)
#
# with open("train_data_process_more",'w') as file:
#     for l in lines:
#         file.write(l+"\n")
#
#
#
# print(positive)
# print(negtive)
#
#
#
#
