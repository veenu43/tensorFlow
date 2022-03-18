import tensorflow as tf
import numpy as np

tf.debugging.set_log_device_placement(False)

# Create Variable
# if no datatype then tensor variable will be drive
v1 = tf.Variable([[1.5, 2, 5], [2, 6, 8]])
print(v1)

v2 = tf.Variable([[1, 2, 5], [2, 6, 8]], dtype=tf.float32)
print(v2)

# Operation output is tensor not variable
print(tf.add(v1, v2))

v3 = tf.Variable([[1.5, 2, 5]])
print(tf.add(v1, v3))

# convert to Tensor
print(tf.convert_to_tensor(v1))

# give numpy representation of an array
print(v1.numpy())

# Variables are mutable unlike tensor
print(v1)
v1.assign([[10, 20, 30], [40, 50, 60]])
print(v1)

# Assignment for specific elements
v1[0, 0].assign(100)
print(v1)

# operations can be mutated
print(v1.assign_add([[1, 1, 1], [1, 1, 1]]))
v6 = v1.assign_sub([[1, 2, 3], [2, 3, 7]])
print(v6)

# Variables can be assign using another variable then copy of the variable is being
# variable dont share memory
var_a = tf.Variable([2.0, 3.0])
var_b = tf.Variable(var_a)
print(var_a)
print(var_b)
# Assigned value to a variable should be compatible
var_b.assign([2.1, 3.1])
print(var_b)
