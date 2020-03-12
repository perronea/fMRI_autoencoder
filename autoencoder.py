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

from keras.models import load_model

from keras.models import Model
from keras.layers import Dense, Input, Layer, InputSpec
from keras.regularizers import l1
from keras.optimizers import Adam
from keras import backend as K
from keras import regularizers, activations, initializers, constraints, Sequential
from keras.constraints import UnitNorm, Constraint

NUM_PARCELS = 352

class DenseTied(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 **kwargs):
        self.tied_to = tied_to
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
                
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = K.transpose(self.tied_to.kernel)
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

class WeightsOrthogonalityConstraint (Constraint):
    def __init__(self, encoding_dim, weightage = 1.0, axis = 0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis
        
    def weights_orthogonality(self, w):
        if(self.axis==1):
            w = K.transpose(w)
        if(self.encoding_dim > 1):
            m = K.dot(K.transpose(w), w) - K.eye(self.encoding_dim)
            return self.weightage * K.sqrt(K.sum(K.square(m)))
        else:
            m = K.sum(w ** 2) - 1.
            return m

    def __call__(self, w):
        return self.weights_orthogonality(w)



def plot_autoencoder_outputs(autoencoder, test_data):
    decoded_pconns = autoencoder.predict(test_data)

    # number of example pconns to show
    n = 3
    fig = plt.figure(figsize=(8,6))
    for i in range(n):
        # plot original pconn
        ax = plt.subplot(2, n, i+1)
        im = ax.imshow(reconstruct_pconn(test_data[i+2]), cmap='hot')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_title('Subject 1')
        if i == 1:
            ax.set_title('Original Parcellated Connectivity Matrix\nSubject 2')
        if i ==2:
            ax.set_title('Subject 3')

        # plot generated pconn
        ax = plt.subplot(2, n, i+1+n)
        im = ax.imshow(reconstruct_pconn(decoded_pconns[i+2]), cmap='hot')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_title('Subject 1')
        if i == 1:
            ax.set_title('Reconstructed Parecellated Connectivity Matrix\nSubject 2')
        if i == 2:
            ax.set_title('Subject 3')
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02)
    cb_ax = fig.add_axes([0.83,0.1,0.02,0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    #fig.colorbar(im)
    plt.savefig('results.png')

#def plot_autoencoder_outputs(autoencoder, test_data):
#    decoded_pconns = autoencoder.predict(test_data)
#    fig, axs = plt.subplots(nrows=2, ncols=2)
#    im1 = axs[0,0].imshow(reconstruct_pconn(test_data[1]), cmap='hot')
#    im2 = axs[0,1].imshow(reconstruct_pconn(test_data[3]), cmap='hot') 
#    im3 = axs[1,0].imshow(reconstruct_pconn(decoded_pconns[1]), cmap='hot')
#    im4 = axs[1,1].imshow(reconstruct_pconn(decoded_pconns[3]), cmap='hot')
#    fig.colorbar()
        

def reconstruct_pconn(pconn_vec):
    pconn_vec=normalize(pconn_vec)
    iu = np.triu_indices(NUM_PARCELS, 1)
    recon_pconn = np.zeros((NUM_PARCELS, NUM_PARCELS))
    np.fill_diagonal(recon_pconn,1)
    recon_pconn[iu] = pconn_vec
    recon_pconn = recon_pconn + recon_pconn.T - np.diag(np.diag(recon_pconn))
    return(recon_pconn)

def normalize(vec):
    return 2*(vec-min(vec))/(max(vec)-min(vec))-1
    

def load_pconn_data(data_dir):
    # Load all pconns in the data directory into a numpy array from a txt
    # All pconns are created from the Gordon parcellation containing 352 parcels (size: 352x352)
    # Get just the lower triangle of the matrix
    
    pconn_list = os.listdir(data_dir)

    # Initialize empty numpy array
    pconn_data = np.empty((len(pconn_list), 61776))

    # indices of the upper triangle
    iu = np.triu_indices(NUM_PARCELS, 1)
    for i, pconn in enumerate(pconn_list):
        pconn_path = os.path.join(data_dir, pconn)
        pconn_mat = np.loadtxt(pconn_path)
        pconn_mat = np.clip(pconn_mat, -1, 1)
        pconn_tri = pconn_mat[iu]
        pconn_data[i] = pconn_tri

    split = 0.1
    test_data = pconn_data[0:int(len(pconn_list) * split)-1]
    train_data = pconn_data[int(len(pconn_list) * split):-1]

    print(train_data.shape)
    print(test_data.shape)
    return (train_data, test_data)

def plot_training(training):

    # Plot training & validation loss values
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('Autoencoder Mean Squared Error')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show() 

def load_autoencoder(h5_path):
    autoencoder = load_model(return_path, compile=False) 


def train_autoencoder(train_data, test_data):

    input_size = 61776
    hidden_size = 432
    hidden2_size = 48
    latent_size = 12

    # Build encoder

    input_pconn = Input(shape=(input_size,))
    d1 = Dense(hidden_size, activation='relu', kernel_regularizer=WeightsOrthogonalityConstraint(hidden_size, weightage=1., axis=0))
    d2 = Dense(hidden2_size, activation='relu', kernel_regularizer=WeightsOrthogonalityConstraint(hidden2_size, weightage=1., axis=0))
    d3 = Dense(latent_size, activation='relu', kernel_regularizer=WeightsOrthogonalityConstraint(latent_size, weightage=1., axis=0))
    hidden_1 = d1(input_pconn)
    hidden2_1 = d2(hidden_1)
    latent = d3(hidden2_1)
    
    encoder = Model(input_pconn, latent, name='encoder')
    encoder.summary()

    # Build decoder

    latent_inputs = Input(shape=(latent_size,), name='decoder_input')
    #hidden2_2 = Dense(hidden2_size, activation='relu')(latent_inputs)
    #hidden_2 = Dense(hidden_size, activation='relu')(hidden2_2)
    #output_pconn = Dense(input_size, activation='sigmoid')(hidden_2)
    td3 = DenseTied(hidden2_size, activation='relu', tied_to=d3)
    td2 = DenseTied(hidden_size, activation='relu', tied_to=d2)
    td1 = DenseTied(input_size, activation='sigmoid', tied_to=d1)
    hidden2_2 = td3(latent_inputs)
    hidden_2 = td2(hidden2_2)
    output_pconn = td1(hidden_2)

    decoder = Model(latent_inputs, output_pconn, name="decoder")
    decoder.summary()

    # Build autoencoder = encoder + decoder
    #autoencoder = Model(input_pconn, output_pconn)
    autoencoder = Model(input_pconn, decoder(encoder(input_pconn)), name='autoencoder')
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    #encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)

    training = autoencoder.fit(train_data, train_data, validation_split=0.1, epochs=20, batch_size=1, verbose=1)

    plot_training(training)


    return (autoencoder, encoder)


def main():

    data_dir = '/home/exacloud/lustre1/fnl_lab/projects/VAE/data/pconn_data'

    (train_data, test_data) = load_pconn_data(data_dir)

    autoencoder, encoder = train_autoencoder(train_data, test_data)
    #autoencoder.save('gordon_pconn_autoencoder.h5')
    #encoder.save('gordon_pconn_encoder.h5')
    
    #autoencoder = load_model('/home/exacloud/lustre1/fnl_lab/projects/VAE/code/gordon_pconn_autoencoder.h5', compile=False) 
     
    plot_autoencoder_outputs(autoencoder, test_data)



if __name__ == "__main__":
    main()


