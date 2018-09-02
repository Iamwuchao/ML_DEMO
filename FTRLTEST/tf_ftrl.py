import tensorflow as tf
import numpy as np
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle

# Define the placeholder
x = tf.placeholder("float", [None, 12568])
y_ = tf.placeholder("float", [None, 1])

# Define the variable of the model
W = tf.Variable(tf.random_uniform([1, 12568], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
#alpha = 0.01

y = tf.sigmoid(tf.matmul(x, tf.transpose(W)) + b)
y_pred =tf.sigmoid(tf.matmul(x, tf.transpose(W)) + b)
# clipping y to avoid log(y) become infinite
y = tf.clip_by_value(y, 1e-10, 1-1e-10)

# Minimize the negative log likelihood.
#loss = tf.reduce_mean(-tf.matmul(tf.transpose(y_), tf.log(y)) - tf.matmul(tf.transpose(1-y_), tf.log(1-y)))
#l1 = tf.contrib.layers.l1_regularizer(alpha)(W)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y)) #+ l1
#loss =tf.reduce_mean(- tf.reshape(y_,[-1, 1]) * tf.log(y) - (1 - tf.reshape(y_,[-1, 1])) * tf.log(1 - y))
#optimizer = tf.train.FtrlOptimizer(0.03, l1_regularization_strength=0.01, l2_regularization_strength=0.01)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
auc = tf.metrics.auc(labels=y_,predictions=y)
# Before starting, initialize the variables.  We will 'run' this first.
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

# # Launch the graph.
sess = tf.Session()
sess.run(init)

x_train, y_train = load_svmlight_file("./train_data_process")
x_train_new, y_train_new = shuffle(x_train, y_train)

for sample_index in range(x_train_new.shape[0]):
    sess.run(train, {x:x_train_new[sample_index].toarray(), y_:np.array([y_train_new[sample_index]]).reshape([1,1])})
    train_W = sess.run(W)
    train_b = sess.run(b)
    if sample_index % 200 == 0:
        size = 1000
        if sample_index+1000 < x_train_new.shape[0]:
            print(sample_index,sess.run(loss / size, {x:x_train_new[sample_index:sample_index+1000].toarray(), y_:np.array([y_train_new[sample_index:sample_index+1000]]).reshape([1000,1])}))

#End print the model and the training accuracy
print('W:', train_W)
print('b:', train_b)

# saver = tf.train.Saver()
# ckpt = tf.train.get_checkpoint_state("./model")
# if ckpt and ckpt.model_checkpoint_path:
#             print("Success to load %s." % ckpt.model_checkpoint_path)
#             saver.restore(sess, ckpt.model_checkpoint_path)
#
x_data,y_data = load_svmlight_file("./test_data_process")
#
# train_W = sess.run(W)
# train_b = sess.run(b)
# print('W:', train_W)
# print('b:', train_b)

y_pre = sess.run(y_pred,feed_dict={x:x_data.toarray(),y_:np.array(y_data).reshape([-1,1])})
auc = metrics.roc_auc_score(y_data.reshape([-1,1]), y_pre)
print(auc)
train_W = sess.run(W)
# w_list = train_W.tolist()
# with open("ftrl_weight",'w') as file1:
#     for w in w_list:
#         file1.write(str(w)+"\n")
# # #predict_accuracy(train_y, y_data)