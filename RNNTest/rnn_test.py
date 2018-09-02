import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/data",one_hot='True')
learning_rate = 0.001
train_steps = 10000
batch_size = 128
display_step = 200

num_input = 28
timesteps = 28
num_hidden = 128
num_classes = 10

X = tf.placeholder("float",[None,timesteps,num_input])
Y = tf.placeholder("float",[num_classes])

weights = {
    'out':tf.Variable(tf.random_normal([num_hidden,num_classes]))
}

biases = {
    'out':tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x,weights,biases):
    x= tf.unstack(x,timesteps,1)


