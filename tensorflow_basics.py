import tensorflow as tf

## VARIABLES AND CONSTANTS
constant1 = tf.constant(3)
constant2 = tf.Variable(5)

# Initialize all variables
init = tf.global_variables_initializer()
product = tf.multiply(constant1, constant2)
with tf.Session() as sess:

    # sess.run(init) runs the initialization
    sess.run(init)
    # Variables are updated using tf.assign
    update = tf.assign(constant2, product)

    # Move symbolic output product into product_result by running the graph
    sess.run(update)
    product_result = sess.run(product)


print product_result


## PLACEHOLDERS

input = tf.placeholder(tf.int8)
multiplier = tf.Variable(5, dtype=tf.int8)

product = tf.multiply(input, multiplier)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Product node in the graph is being called
    result = sess.run(product, feed_dict={input: 10})
    print result

