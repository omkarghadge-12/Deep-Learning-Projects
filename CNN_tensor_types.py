import tensorflow as tf

# Scaler tensor(0D)
scaler_tensor = tf.constant(11)
print("Scaler Tensor : ",scaler_tensor)

# 1D tensor(vector)
vector_tensor = tf.constant([11,21,51,101])
print("1D Tensor(vector) : ",vector_tensor)

# 2D tensor(Matrix)
matrix_tensor = tf.constant([[10,20,30],[40,50,60]])
print("2D tensor(Matrix) : ",matrix_tensor)

# 3D tensor
tensor_3D = tf.constant([
    [[1,2],[3,4]],
    [[5,5],[6,6]],
    [[7,8],[9,10]]
])
print("3D tensor : ",tensor_3D)