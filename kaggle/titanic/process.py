import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
#from titanic.process_data import *
from tensorflow.python import debug as tf_debug
import pandas as pd

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
k=4  #FM 的超参

data = process_data(train_file)

train_X = data.loc[:,X_feature]
print(train_X.shape)
train_X_list = []
for index,row in train_X.iterrows():
    new_dict_row = {}
    for key in X_feature:
        new_dict_row[key] = row[key]
    train_X_list.append(new_dict_row)

v = DictVectorizer()
print(len(train_X_list))
train_X_onehot =v.fit_transform(train_X_list).toarray()
print("train_X_onehot")
print(type(train_X_onehot))
#train_X_onehot = np.array(train_X_onehot)
#print(train_X_onehot.shape)

train_Y = data.loc[:,['Survived']].values
print(type(train_Y))
print(train_Y.shape)
#print(train_Y.shape)
#train_Y = np.array(train_Y)
n_samples = train_X.shape[0]

# X = tf.placeholder("float")
# Y = tf.placeholder("float")

w0 = tf.Variable(0.1)
w1 = tf.Variable(tf.truncated_normal([n]))
w2 = tf.Variable(tf.truncated_normal([n,k]))

x_ = tf.placeholder(tf.float32,[None,n])
y_ = tf.placeholder(tf.float32,[None,1])

batch = tf.placeholder(tf.int32)

w2_new = tf.reshape(tf.tile(w2,[891,1]),[-1,n,k])

board_x = tf.reshape(tf.tile(x_,[1,k]),[-1,n,k])

board_x2 = tf.square(board_x)

q = tf.square(tf.reduce_sum(tf.multiply(w2_new,board_x),axis=1))
h = tf.reduce_sum(tf.multiply(tf.square(w2_new),board_x2),axis=1)

y_fm = w0 + tf.reduce_sum(tf.multiply(x_,w1),axis=1) +\
       1/2*tf.reduce_sum(q-h,axis=1)
#pred = tf.nn.sigmoid(y_fm)


#pred = tf.nn.softmax(y_fm)
#tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=pred)#
cost = tf.reduce_sum(0.5*tf.square(y_fm-y_))#tf.reduce_mean(-tf.reduce_sum(y_*tf.log(pred)))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    for epoch in range(10):
        sess.run(optimizer, feed_dict={x_: train_X_onehot, y_: train_Y,batch: 30})
    print(sess.run(cost,feed_dict={x_: train_X_onehot, y_: train_Y,batch: 30}))
    #print(sess.run(accury, feed_dict={x_: x_test, y_: y_test, batch: 30}))








