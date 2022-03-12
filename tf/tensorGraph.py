import tensorflow as tf

# Constant are immutable
# Specify the operations and the data
a = tf.constant(6, name='constant_a')
b = tf.constant(3, name='constant_b')
c = tf.constant(10, name='constant_c')
d = tf.constant(5, name='constant_d')

mul = tf.multiply(a, b, name="mul")
print("Multiply: ", mul)

div = tf.divide(c, d, name="div1")
print("Divided: ", div)

addn = tf.add_n([mul, tf.cast(div, tf.int32)], name="addn")
print("addn: ", addn)

'''
tf.function(addn)
sess = tf.compat.v1.Session()
sess.run(addn)
'''
