import tensorflow as tf
import numpy as np

# This will print out where tensor are located and which device are executed
tf.debugging.set_log_device_placement(True)

print(tf.executing_eagerly())

x0 = tf.constant(3)
print(x0)

# print tensor each attributes
print(f"Shape {x0.shape}, dataType {x0.dtype}, value {x0.numpy()}")

result0 = x0 + 5
print(result0)

x1 = tf.constant([1.1, 2.2, 3.3, 4.4])
print(x1)

result1 = x1 + 5
print(result1)

# tensor are immutable.Every operation creates new tensor
# ad via tensor api
result1 = tf.add(x1, tf.constant(5.0))
print(result1)

# Create tensor 2 dimensional array
x2 = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
print(x2)

# Cast from into to float
x2 = tf.cast(x2, tf.float32)
print(x2)

# mutiplication for same type
result3 = tf.multiply(x1, x2)
print(result3)

x3 = tf.constant([[1.1, 2.2, 3.3, 4.4], [2, 1, 2, 3]])
result4 = tf.multiply(x3, x2)
print("numpy", result4.numpy())

# numpy array
arr_x1 = np.array([[10,20],[30,40],[50,60]])
print(arr_x1)

# Convert numpy array into tensor
x5 = tf.convert_to_tensor(arr_x1)
print(x5)
