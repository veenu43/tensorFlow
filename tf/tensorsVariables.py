import tensorflow as tf
import numpy as np

tf.debugging.set_log_device_placement(True)

# Variables are mutable unlike tensor

# Create Variable
# if no datatype then tensor variable will infe
v1 = tf.Variable([[1.5, 2, 5], [2, 6, 8]])
print(v1)

v2 = tf.Variable([[1, 2, 5], [2, 6, 8]], dtype=tf.float64)
print(v2)

print(tf.add(v1,v2))