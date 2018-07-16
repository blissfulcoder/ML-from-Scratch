import numpy as np
from matplotplib import pyplot as plt
import pandas as pd

import keras
from keras.layers import Dense, Activation, Input
from keras.models import Model
ds = pd.read_csv("D:\ML--perceptron\mnist\train.csv")
data = ds.values
print (data.head())

X_data = data[:, 1:]
X_std = X_data/255.0

n_train = int(0.75*X_std.shape[0])
n_val = int(0.25*X_std.shape[0])

X_train = X_std[:n_train]
X_val = X_std[n_train:n_train+n_val]

print (X_train.shape, X_val.shape)