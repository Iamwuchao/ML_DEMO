import tensorflow as tf

aa = tf.Variable(tf.fill([3,3,4],1))
b = tf.Variable(tf.fill([4],2))
c = tf.add(aa,b)

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   value = sess.run(c)
   print(value)