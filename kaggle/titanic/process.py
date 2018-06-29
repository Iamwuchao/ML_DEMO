import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
#from titanic.process_data import *
from tensorflow.python import debug as tfdbg
import pandas as pd
import xgboost as xgb
import sklearn

train_file = "train.csv"
test_file = "test.csv"

feature_list = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']

def process_data(file):
    df = pd.read_csv(file)
    #print(df.shape)
    feature_df = df.loc[:,feature_list]
    #print(feature_df.shape)
    return feature_df

X_feature = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
learning_rate = 0.01
training_epochs = 100
display_step = 50
n=159  #X的长度
k=10  #FM 的超参

data1 = process_data(train_file)

data2 = process_data(test_file)

data = np.concatenate((data1,data2),axis=0)

X = data.loc[:,X_feature]
print(X.shape)
X_list = []
for index,row in X.iterrows():
    new_dict_row = {}
    for key in X_feature:
        new_dict_row[key] = row[key]
    X_list.append(new_dict_row)

v = DictVectorizer()
print(len(X_list))
X_onehot =v.fit_transform(X_list).toarray()
where_are_nan = np.isnan(X_onehot)
X_onehot[where_are_nan] = 0
print("train_X_onehot")
print(type(X_onehot))
#train_X_onehot = np.array(train_X_onehot)
print(X_onehot.shape)

train_Y = data.loc[:,['Survived']].values

print(type(train_Y))
print(train_Y.shape)

dtrain = xgb.DMatrix(data=train_X_onehot,label=train_Y)

params={'booster':'gbtree',
	    'objective': 'binary:logistic',
	    'eval_metric':'logloss',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }


num_round = 200
bst = xgb.train(params, dtrain, num_boost_round=2000)
preds = bst.predict(xgb.DMatrix(train_X_onehot))
print(type(preds))

for p in preds:
    print(p)

#ppp = (preds.round()+1)%2


acc_val = sklearn.metrics.accuracy_score(train_Y, preds.round())
print("#######")
print(acc_val)
# print(type(train_Y))
# print(train_Y.shape)
# #print(train_Y.shape)
# #train_Y = np.array(train_Y)
# n_samples = train_X.shape[0]
#
# # X = tf.placeholder("float")
# # Y = tf.placeholder("float")
#
# w0 = tf.Variable(0.1,name="w0")
# w1 = tf.Variable(tf.truncated_normal([n]),name='w1')
# w2 = tf.Variable(tf.truncated_normal([n,k],name='w2'))
#
# x_ = tf.placeholder(tf.float32,[None,n],name='x_')
# y_ = tf.placeholder(tf.float32,[None,1],name='y_')
#
# batch = tf.placeholder(tf.int32,name='batch')
#
# w2_new = tf.reshape(tf.tile(w2,[batch,1]),[-1,n,k],name='w2_new')
#
# board_x = tf.reshape(tf.tile(x_,[1,k]),[-1,n,k],name='board_x')
#
# board_x2 = tf.square(board_x,name='board_x2')
#
# q = tf.square(tf.reduce_sum(tf.multiply(w2_new,board_x),axis=1),name='q')
# h = tf.reduce_sum(tf.multiply(tf.square(w2_new),board_x2),axis=1,name='h')
#
# tem_x1 = tf.reduce_sum(tf.multiply(x_,w1),axis=1,name='tem_x1')
# tem_x2 = 0.5*tf.reduce_sum(q-h,axis=1)
# y_fm = w0 + tem_x1 + tem_x2
#
# pred = tf.nn.sigmoid(y_fm)
# pred = tf.reshape(pred,[-1,1])
# pred_new = tf.round(pred)
#
#
# cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=pred)) +\
# tf.contrib.layers.l1_regularizer(0.01)(w1) + tf.contrib.layers.l2_regularizer(0.1)(w2)#
# #cost = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(pred), reduction_indices=1))
# #cost = tf.reduce_sum(0.5*tf.square(y_fm-y_),name='cost')#tf.reduce_mean(-tf.reduce_sum(y_*tf.log(pred)))
#
# optimizer = tf.train.AdagradOptimizer(learning_rate=0.2).minimize(cost)#MomentumOptimizer(learning_rate=0.01,momentum=0.5).minimize(cost)
#     #AdagradOptimizer(learning_rate=0.2).minimize(cost)
#
# correct_prediction = tf.equal(y_,pred_new)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     # sess = tfdbg.LocalCLIDebugWrapperSession(sess)
#     # sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)
#     sess.run(init)
#     sess.run(tf.local_variables_initializer())
#     start=0
#     end = 100
#     for epoch in range(1,2000):
#         # x1 = tf.train.batch(train_X_onehot,batch_size=10)
#         # y1 = tf.train.batch(train_Y,batch_size=10)
#         sess.run(optimizer, feed_dict={x_: train_X_onehot[start:end], y_: train_Y[start:end],batch: 100})
#         print(sess.run(accuracy,feed_dict={x_: train_X_onehot, y_: train_Y,batch: 891}))
#         start+=100
#         end+=100
#         if end>891:
#             start=random.randint(0,9)
#             end=start +100
#         #print(sess.run(tf.Print(pred_new,[pred_new]),feed_dict={x_: train_X_onehot, y_: train_Y,batch: 30}))
#         #print(sess.run(accury, feed_dict={x_: x_test, y_: y_test, batch: 30}))








