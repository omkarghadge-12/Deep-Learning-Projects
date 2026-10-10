import tensorflow as tf

tensor1 = tf.constant([10,20,30])
tensor2 = tf.constant([1,2,3])

addition = tf.add(tensor1 , tensor2)
print("Addition is : ",addition)

subtraction = tf.subtract(tensor1 , tensor2)
print("Subtraction is : ",subtraction)

multiplication = tf.multiply(tensor1 , tensor2)
print("Multiplication is : ",multiplication)

division = tf.divide(tensor1 , tensor2)
print("Division is : ",division)

squ = tf.square(tensor1)
print("Square is : ",squ)

Sum = tf.reduce_sum(tensor1)
print("Reduce sum : ",Sum)

Mean = tf.reduce_mean(tensor1)
print("Mean is : ",Mean)
