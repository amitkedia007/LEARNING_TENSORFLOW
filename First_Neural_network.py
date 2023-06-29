import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape , y_train.shape)
