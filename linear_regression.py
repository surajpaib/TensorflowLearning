import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Defining the input and output training data
x_train = np.random.rand(100).astype(np.float32)
y_train = 5 * x_train + 10
y_train = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.2))(y_train)
print x_train, y_train
plt.plot(x_train, y_train, 'ro')
plt.show()


# We need to predict y using value of x_train. a and b are variables since we
# will be updating their values every iteration. They are also the variables whose values we need to unlock
x = tf.placeholder(tf.float32)
y_actual = tf.placeholder(tf.float32)
a = tf.Variable(0.05, dtype=tf.float32)
b = tf.Variable(0.02, dtype=tf.float32)
y = a * x + b
init = tf.global_variables_initializer()
# Defined MSE, Optimizer and Training objective
mse = tf.reduce_mean(tf.square(y - y_actual))
optimizer = tf.train.GradientDescentOptimizer(0.5, name='GD')
train = optimizer.minimize(mse)


with tf.Session() as sess:
    sess.run(init)
    for step in range(1, 1000):
        outputs = sess.run([train, a, b], feed_dict={x:x_train, y_actual: y_train})[1:]
        if step % 50 == 0:
            print step, outputs

    # Assign final values to a and b
    update_a = tf.assign(a, outputs[0])
    update_b = tf.assign(b, outputs[1])
    sess.run([update_a, update_b])
    a, b = sess.run([a, b])
    print a, b

# Plot line between the points
plt.plot(x_train, y_train, 'ro')
plt.plot(x_train, a * x_train + b)
plt.show()




