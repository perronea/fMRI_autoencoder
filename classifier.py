#! /usr/bin/env python3

#### Steps


import os
import numpy as np
import matplotlib.pyplot as plt
import random

from keras.models import load_model

from keras.models import Model
from keras.layers import Dense, Input
from keras.regularizers import l1
from keras.optimizers import Adam

from keras.models import Sequential

NUM_PARCELS = 352


def reconstruct_pconn(pconn_vec):
    iu = np.triu_indices(NUM_PARCELS, 1)
    recon_pconn = np.zeros((NUM_PARCELS, NUM_PARCELS))
    np.fill_diagonal(recon_pconn,1)
    recon_pconn[iu] = pconn_vec
    recon_pconn = recon_pconn + recon_pconn.T - np.diag(np.diag(recon_pconn))
    return recon_pconn


def load_and_encode(model_path, train_pconns, test_pconns):
    encoder = load_model(model_path, compile=False)
    coded_train_pconns = encoder.predict(train_pconns)
    coded_test_pconns = encoder.predict(test_pconns)

    return (coded_train_pconns, coded_test_pconns)

def plot_training(training):
    # Plot training & validation accuracy values
    plt.plot(training.history['acc'])
    plt.plot(training.history['val_acc'])
    plt.title('Classifier Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('Classifier Binary Crossentropy')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def train_classifier(coded_train_pconns, train_labels, coded_test_pconns, test_labels):
    
    classifier = Sequential()
    classifier.add(Dense(24, input_dim=12, activation='relu'))
    classifier.add(Dense(12, activation='relu'))
    classifier.add(Dense(8, activation='relu'))
    classifier.add(Dense(1, activation='sigmoid'))
    
    classifier.summary()

    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    training = classifier.fit(coded_train_pconns, train_labels, validation_split=0.2, epochs=20, batch_size=10)

    plot_training(training)

    _, accuracy = classifier.evaluate(coded_test_pconns, test_labels)

    print('Accuracy: %.2f' % (accuracy*100))
    return
    

def load_class_data():
    # Load all pconns and assign ASD subjects to class 1 and non-ASD subjects to class 0

    asd_dir = '/home/exacloud/lustre1/fnl_lab/projects/VAE/data/asd'
    adhd_dir = '/home/exacloud/lustre1/fnl_lab/projects/VAE/data/adhd'

    pconn_list = []
    for pconn in os.listdir(asd_dir):
        full_path = os.path.join(asd_dir, pconn)
        pconn_list.append(full_path)
    for pconn in os.listdir(adhd_dir):
        full_path = os.path.join(adhd_dir, pconn)
        pconn_list.append(full_path)

    labels = np.zeros(len(pconn_list))
    labels[0:len(os.listdir(asd_dir))] = 1

    pconn_ix = list(range(0,len(pconn_list)))
    random.shuffle(pconn_ix)

    # Initialize empty numpy array
    pconn_data = np.empty((len(pconn_list), 61776))

    # indices of the upper triangle
    iu = np.triu_indices(NUM_PARCELS, 1)
    for i, pconn_path in enumerate(pconn_list):
        pconn_mat = np.loadtxt(pconn_path)
        np.clip(pconn_mat, -1, 1)
        pconn_tri = pconn_mat[iu]
        pconn_data[i] = pconn_tri

    split = 0.1
    test_data_ix = pconn_ix[0:int(len(pconn_ix) * split)-1]
    train_data_ix = pconn_ix[int(len(pconn_ix) * split):-1]

    test_pconns = pconn_data[test_data_ix]
    train_pconns = pconn_data[train_data_ix]
    test_labels = labels[test_data_ix]
    train_labels = labels[train_data_ix]

    print(train_pconns.shape)
    print(test_pconns.shape)
    print(train_labels.shape)
    print(test_labels.shape)
    return (train_pconns, train_labels, test_pconns, test_labels)


def main():


    (train_pconns, train_labels, test_pconns, test_labels) = load_class_data()

    (coded_train_pconns, coded_test_pconns) = load_and_encode('/home/exacloud/lustre1/fnl_lab/projects/VAE/code/gordon_pconn_encoder.h5', train_pconns, test_pconns)
    print(coded_train_pconns.shape)
    print(coded_test_pconns.shape)

    train_classifier(coded_train_pconns, train_labels, coded_test_pconns, test_labels)

    return

    
if __name__ == "__main__":
    main()

