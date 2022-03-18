import tensorflow as tf

print("executing_eagerly: ", tf.executing_eagerly())
# This will print out where tensor are located and which device are executed
tf.debugging.set_log_device_placement(False)
a = tf.constant(5, name="a")
b = tf.constant(7, name="b")
c = tf.add(a, b, name="sum")
print("c",c)

# This will give error when eager execution mode is ON
# Disable eager execution
tf.compat.v1.disable_eager_execution()
print("executing_eagerly: ", tf.compat.v1.executing_eagerly())

# reset default graph
# All operations are added in default graph and gets executed when sess.run is called
tf.compat.v1.reset_default_graph()
a = tf.constant(5, name="a")
b = tf.constant(7, name="b")
c = tf.add(a, b, name="sum")
sess = tf.compat.v1.Session()

# Value not available till computation graph is not executed
print("Before execution", c)
print(sess.run(c))
print("after execution", c)

d = tf.multiply(a, b, name="product")
print("Before execution", d)
print(sess.run(d))


# Operations on mutable parameters like variables are not available till computation graph are executed
m = tf.Variable([[1, 2, 5], [2, 6, 8]], dtype=tf.float32, name='m')
c = tf.Variable([[6, 72, 2], [5, 8, 78]], dtype=tf.float32, name='c')
print(m)
print(c)

# Value needs to specially initialized in tensor session , to be visible
# trainable parameters were variable in tensor 1 and Input parameters  provided to computation graph were placeholder
x = tf.compat.v1.placeholder(tf.float32, shape=[2, 3], name='x')

# Actual parameter is provided are run time to computation graph
print("X", x)
y = m * x + c
print(y)

#
init = tf.compat.v1.global_variables_initializer()
# !rm -rf ./logs/

with tf.compat.v1.Session() as sess:
    sess.run(init)
    y_output = sess.run(y, feed_dict={x: [[100.0, 100.0, 100.0], [5, 8, 78]]})
    print("Final result: mx+c = ", y_output)
    writer = tf.compat.v1.summary.FileWriter('./logs', sess.graph)
    writer.close()


sess.close()
