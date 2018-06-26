
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

n_hidden_1 = 256
n_hidden_2 = 256
num_input = 784
num_classes = 10

X = tf.placeholder("float",[None,num_input])
Y = tf.placeholder("float",[None,num_classes])

weights = {

}