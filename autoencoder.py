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

num_parcels = 352

def plot_autoencoder_outputs(autoencoder):
    decoded_pconns = autoencoder.predict(test_data)

    # number of example pconns to show
    n = 5
    plt.figure(figsize=(10, 4.5))
    for i in range(n):
        # plot original pconn
        ax = plt.subplot(2, n, i+1)
        plt.imshow(reconstruct_pconn(test_data[i]), cmap='hot')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Original pconn')

        # plot generated pconn
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(reconstruct_pconn(decoded_pconns[i]), cmap='hot')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Reconstructed pconn')
    plt.savefig('results.png')
        

def reconstruct_pconn(pconn_vec):
    iu = np.triu_indices(num_parcels, 1)
    recon_pconn = np.zeros((num_parcels, num_parcels))
    np.fill_diagonal(recon_pconn,1)
    recon_pconn[iu] = pconn_vec
    recon_pconn = recon_pconn + recon_pconn.T - np.diag(np.diag(recon_pconn))
    return(recon_pconn)

def load_data():
    # Load all pconns in the data directory into a numpy array from a txt
    # All pconns are created from the Gordon parcellation containing 352 parcels (size: 352x352)
    # Get just the lower triangle of the matrix
    
    data_dir = '/home/exacloud/lustre1/fnl_lab/projects/VAE/data/pconn_data'
    pconn_list = os.listdir(data_dir)

    # Initialize empty numpy array
    pconn_data = np.empty((len(pconn_list), 61776))

    # indices of the upper triangle
    iu = np.triu_indices(num_parcels, 1)
    for i, pconn in enumerate(pconn_list):
        pconn_path = os.path.join(data_dir, pconn)
        pconn_mat = np.loadtxt(pconn_path)
        pconn_tri = pconn_mat[iu]
        pconn_data[i] = pconn_tri

    # load numpy data into tensorflow

    split = 0.1
    test_data = pconn_data[0:int(len(pconn_list) * split)-1]
    train_data = pconn_data[int(len(pconn_list) * split):-1]

    print(train_data.shape)
    print(test_data.shape)
    return (train_data, test_data)

(train_data, test_data) = load_data()


input_size = 61776
hidden_size = 432
hidden2_size = 48
code_size = 12

input_pconn = Input(shape=(input_size,))
hidden_1 = Dense(hidden_size, activation='relu')(input_pconn)
hidden2_1 = Dense(hidden2_size, activation='relu')(hidden_1)
code = Dense(code_size, activation='relu')(hidden2_1)
hidden2_2 = Dense(hidden2_size, activation='relu')(code)
hidden_2 = Dense(hidden_size, activation='relu')(hidden2_2)
output_pconn = Dense(input_size, activation='sigmoid')(hidden_2)

autoencoder = Model(input_pconn, output_pconn)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(train_data, train_data, epochs=10)


plot_autoencoder_outputs(autoencoder)





