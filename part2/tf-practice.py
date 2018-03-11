import numpy as np
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

coefficients = np.array([[1.], [-10], [25.]])

w = tf.Variable([0], dtype=tf.float32)
x = tf.placeholder(tf.float32, [3, 1])

cost = w ** 2 - 10 * w + 25

train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)
print(session.run(w))

session.run(train, feed_dict={x: coefficients})
print(session.run(w))

for i in range(100):
    session.run(train, feed_dict={x: coefficients})
    print(i, ":", session.run(w))

print(session.run(w))

session.close()
