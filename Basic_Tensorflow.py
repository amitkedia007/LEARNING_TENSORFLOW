# Learning about the manipulaitons of array in tensorflow 
import tensorflow as tf


#single constant value
x = tf.constant(4, shape=(1,1), dtype=tf.float32)

print(x)

#rank-2 tensor

y= tf.constant([[1,2,3,4], [4,5,6,7]])
print(y)
a= tf.ones((3,3))
print(a)
b= tf.zeros((3,3))

# Random matrix creation

c = tf.random.normal((3,3), mean = 0.0, stddev= 1)   #Normal gives you normally distributed random numbers 
print(c)

# multiplication, division
d= tf.constant([1,2,3,4])
e = tf.constant([5,6,7,8])

f= tf.multiply(d, e)  # Similary for division, addition, subtraction we have the same function: add, subtract and multiply
f = d*e
print(f)
g= tf.tensordot(d,e, axes=1) # axes is need to be given to start the multiplication operation for dot product
print(g)


h = tf.random.normal((2,3))
i=  tf.random.normal((3,4))

j = tf.matmul(h,i) # matrix multiplication 
j = h @ i 
print(j)

matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[5., 6.], [7., 8.]])

result = tf.divide(matrix1, matrix2) # matrix division
print(result)

# Slicing and indexing

mydf = tf.constant([[1., 2.,6. , 8.], [3., 4.,5. , 6.]]) 
print("mydf: ",mydf[:,0])

slice = tf.slice(mydf, [0,2],[0,1])
print("slice: ",slice)