#! /usr/bin/env python3

#### Steps


import os
import numpy as np
import matplotlib.pyplot as plt
import random

from keras.models import load_model

#from keras.models import Model
#from keras.layers import Dense, Input
#from keras.regularizers import l1
#from keras.optimizers import Adam
#from keras.models import Sequential

NUM_PARCELS = 352


def reconstruct_pconn(pconn_vec):
    iu = np.triu_indices(NUM_PARCELS, 1)
    recon_pconn = np.zeros((NUM_PARCELS, NUM_PARCELS))
    np.fill_diagonal(recon_pconn,1)
    recon_pconn[iu] = pconn_vec
    recon_pconn = recon_pconn + recon_pconn.T - np.diag(np.diag(recon_pconn))
    return recon_pconn


def load_and_encode_class_data(class_path, encoder):
    pconn_list = []
    for pconn in os.listdir(class_path):
        full_path = os.path.join(class_path, pconn)
        pconn_list.append(full_path)
    
    # Initialize empty numpy array
    pconn_data = np.empty((len(pconn_list), 61776))

    # indices of the upper triangle
    iu = np.triu_indices(NUM_PARCELS, 1)
    for i, pconn_path in enumerate(pconn_list):
        pconn_mat = np.loadtxt(pconn_path)
        np.clip(pconn_mat, -1, 1)
        pconn_tri = pconn_mat[iu]
        pconn_data[i] = pconn_tri

    encoded_pconns = encoder.predict(pconn_data)

    return encoded_pconns
    


def main():

    model_path = '/home/exacloud/lustre1/fnl_lab/projects/VAE/code/gordon_pconn_encoder.h5'
    encoder = load_model(model_path, compile=False)

    encoded_asd = load_and_encode_class_data('/home/exacloud/lustre1/fnl_lab/projects/VAE/data/asd', encoder)
    encoded_adhd = load_and_encode_class_data('/home/exacloud/lustre1/fnl_lab/projects/VAE/data/adhd', encoder)

    avg_encoded_asd = np.var(encoded_asd, axis=0)
    avg_encoded_adhd = np.var(encoded_adhd, axis=0)


    #avg_encoded_asd = np.array([[0.0000000e+00, 0.0000000e+00, 2.1093197e+00, 0.0000000e+00, 7.6074608e+01, 0.0000000e+00, 8.7610809e+01, 0.0000000e+00, 4.5333397e+01, 2.7765019e-02, 0.0000000e+00, 7.2490078e-01]])
    #avg_encoded_adhd = np.array([[0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 7.3685242e+01, 2.8674837e-02, 8.3660789e+01, 0.0000000e+00, 4.0535900e+01, 0.0000000e+00, 0.0000000e+00, 4.5581803e-02]])

    np.set_printoptions(suppress=True)
    print(avg_encoded_asd)
    print(avg_encoded_adhd)

    #avg_encoded_classes = np.concatenate((avg_encoded_asd, avg_encoded_adhd), axis=0)
    avg_encoded_classes = np.array((avg_encoded_asd, avg_encoded_adhd))
    classes = ["Average ASD", "Average ADHD"]
    

    fig, ax = plt.subplots()
    im = ax.imshow(avg_encoded_classes)
    
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)

    cbar = ax.figure.colorbar(im, ax=ax)
    
    plt.show()



    return

    
if __name__ == "__main__":
    main()

