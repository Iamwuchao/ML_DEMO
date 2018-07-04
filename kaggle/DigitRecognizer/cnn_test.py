import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
#from DigitRecognizer import input_data
from tensorflow.python import debug as tfdbg

train_file = "./all/train.csv"
test_file = "./all/test.csv"

def read_train_data(file):
    data = pd.read_csv(file)
    labels = data.loc[:,["label"]].values
    labels = OneHotEncoder().fit_transform(X=labels)
    print("labels shape")
    print(labels.shape)
    x = data.iloc[:,0:-1].values
    return x,labels.toarray()


def read_test_data(file):
    data = pd.read_csv(file)
    return data.values

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')



x = tf.placeholder("float",shape=[None,784],name='x')
y_ = tf.placeholder("float",shape=[None,10],name='y_')

x_image = tf.reshape(x,[-1,28,28,1],name="x_image")

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1,name='h_conv1')
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2,name='h_conv2')
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64],name="h_pool2_flat")
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1,name="h_fc1")

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 =  bias_variable([10])

y_conv = tf.matmul(h_fc1_drop,W_fc2)+b_fc2

y_softmax = tf.nn.softmax(y_conv)
predict = tf.argmax(y_softmax,1,name="predict")
cross_entropy =  tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y_conv)) #-tf.reduce_sum(y_*tf.log(y_conv),name="cross_entropy")
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1),name="correct_prediction")
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

x_train,y_train = read_train_data(train_file)
# print(x_train)
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    # for i in range(20000):
    #     batch = mnist.train.next_batch(50)
    #     print("type")
    #     print(type(batch[0]))
    #     print(batch[0].shape)
    #     print(type(batch[1]))
    #     print(batch[1].shape)
    #     print(batch[0].dtype)
    #     print(batch[1].dtype)
    #     if i % 100 == 0:
    #         test = cross_entropy.eval(feed_dict={
    #             x: batch[0], y_: batch[1], keep_prob: 1.0})
    #         print("step %d, training accuracy %g" % (i, test))
    #     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    #
    # print("test accuracy %g" % accuracy.eval(feed_dict={ \
    #             x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    # sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)
    sess.run(tf.global_variables_initializer())
    x_train,y_train = read_train_data(train_file)
    start = 0
    end = 50
    for i in range(20000):
        #print("start end "+str(start)+" %% "+str(end)+" &&& "+str(x_train.shape[0]))
        batch_x = x_train[start:end]
        batch_y = y_train[start:end]
        batch_x = batch_x.astype(np.float64)
        # print("type")
        # print(type(batch_x))
        # print(batch_x.shape)
        # print(type(batch_y))
        # print(batch_y.shape)
        # print(batch_y.dtype)
        # print(batch_x.dtype)
        start+=50
        end+=50
        if(end>x_train.shape[0]):
            end = 50
            start = 0
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
        if(i%100==0):
            cross_entropy_value = sess.run(cross_entropy,feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
            print("cross_entropy "+str(cross_entropy_value))
            #print("test accuracy %g" % accuracy.eval(feed_dict={x: x_train[0:1000].astype(np.float64), y_:y_train[0:1000], keep_prob: 1.0}))
    saver.save(sess,"./model/"+"CNN_model.ckpt")
    x_test = read_test_data(test_file)
    print("predict")
    result = sess.run(predict,feed_dict={x:x_test.astype(np.float64),y_:y_train,keep_prob:1.0})
    index_image = np.arange(start=1,stop=result.shape[0]+1)
    submission = pd.DataFrame({"ImageId":index_image, "Label":result })
    submission.to_csv("digitRecognizer_submission.csv", index=False)
    #
# with tf.Session() as sess:
#     new_saver = tf.train.import_meta_graph("./model/CNN_model.ckpt.meta")
#     new_saver.restore(sess,tf.train.latest_checkpoint('./model'))
#     predict1 = sess.graph.get_tensor_by_name("predict:0")
#     #correct_prediction = sess.graph.get_operation_by_name("correct_prediction")
#     sess.run(tf.global_variables_initializer())
#     x_test = read_test_data(test_file)
#     print(x_test.shape[0])
#     result = sess.run(predict1,feed_dict={x:x_train[0:1000].astype(np.float64),y_:y_train[0:1000],keep_prob:1.0})
#     print(result)
    # index_image = np.arange(start=1, stop=result.shape[0]+1)
    # submission = pd.DataFrame({"ImageId": index_image, "Label": result})
    # submission.to_csv("digitRecognizer_submission.csv", index=False)


#index_image = np.arange(start=1,stop=result.shape[0])






