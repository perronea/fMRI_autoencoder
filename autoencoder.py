#! /usr/bin/env python3

##########################

# "What I cannot create I do not understand" - Richard Feynmen

##########################


#### Steps

# 1) Load parcelated connectivity matrix
# 2) Take the lower half of the connectivity matrix
# 3) Vectorize the pconn
# 4) Input the vectorized pconn into an encoder consisting of a fully connected neural network with n layers
# 5) Create a decoder
# 6) Recreate the original pconn and visualize

import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.regularizers import l1
from keras.optimizers import Adam


def load_data():
    # Load all pconns in the data directory into a numpy array from a txt
    # All pconns are created from the Gordon parcellation containing 352 parcels (size: 352x352)
    # Get just the lower triangle of the matrix
    
    data_dir = '/home/exacloud/lustre1/fnl_lab/projects/VAE/data/test_pconn_data'
    pconn_list = os.listdir(data_dir)

    # Initialize empty numpy array
    pconn_data = np.empty((len(pconn_list), 61776))

    # indices of the upper triangle
    iu = np.triu_indices(352,1)
    for i, pconn in enumerate(pconn_list):
        pconn_path = os.path.join(data_dir, pconn)
        pconn_mat = np.loadtxt(pconn_path)
        pconn_tri = pconn_mat[iu]
        pconn_data[i] = pconn_tri

    # load numpy data into tensorflow

    split = 0.2
    test_data = pconn_data[0:int(len(pconn_list) * split)-1]
    train_data = pconn_data[int(len(pconn_list) * split):-1]

    print(train_data.shape)
    print(test_data.shape)
    return (train_data, test_data)

(train_data, test_data) = load_data()


input_size = 61776
hidden_size = 432
code_size = 12

input_pconn = Input(shape=(input_size,))
hidden_1 = Dense(hidden_size, activation='relu')(input_pconn)
code = Dense(code_size, activation='relu')(hidden_1)
hidden_2 = Dense(hidden_size, activation='relu')(code)
output_pconn = Dense(input_size, activation='sigmoid')(hidden_2)

autoencoder = Model(input_pconn, output_pconn)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(train_data, train_data, epochs=10)

