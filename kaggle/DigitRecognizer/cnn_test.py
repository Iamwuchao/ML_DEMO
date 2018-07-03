import pandas as pd
import tensorflow as tf

train_file = "./all/train.csv"
test_file = "./all/test.csv"

def read_train_data(file):
    data = pd.read_csv(file)
    labels = data.loc[:,["label"]].values
    x = data.iloc[:,0:-1].values
    print("########")
    print(x.shape)
    x_iamge = tf.reshape(x,[-1,28,28,1])
    return x_iamge,labels


def read_test_data(file):
    data = pd.read_csv(file)
    x_iamge = tf.reshape(data.values, [-1, 28, 28, 1])
    return x_image

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



x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])

x_image = tf.reshape(x,[-1,28,28,1])

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])


h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 =  bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
with tf.Session() as sess:
    # for i in range(20000):
    #     batch = mnist.train.next_batch(50)
    #     if i % 100 == 0:
    #         train_accuracy = accuracy.eval(feed_dict={
    #             x: batch[0], y_: batch[1], keep_prob: 1.0})
    #         print("step %d, training accuracy %g" % (i, train_accuracy))
    #         train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    #
    #         print("test accuracy %g" % accuracy.eval(feed_dict={ \
    #             x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    sess.run(tf.global_variables_initializer())
    x_train,y_train = read_train_data(train_file)
    start = 0
    end = 50
    for i in range(20000):
        batch_x = x_train[start:end]
        batch_y = y_train[start:end]
        start+=50
        end+=50
        if(end>x.shape[0]):
            end = x.shape[0]
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
        if(i%100==0):
            cross_entropy = cross_entropy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
            print("cross_entropy %d\n",cross_entropy)








