#!/usr/bin/env python

from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader


input_size = 61776
l1_size = 432
l2_size = 48
latent_size = 12

SPLIT = 0.3
NUM_PARCELS = 352

num_epochs = 5
batch_size = 2


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, l1_size),
            nn.ReLU(True),
            nn.Linear(l1_size, l2_size),
            nn.ReLU(True),
            nn.Linear(l2_size, latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, l2_size),
            nn.ReLU(True),
            nn.Linear(l2_size, l1_size),
            nn.ReLU(True),
            nn.Linear(l1_size, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder(dataloader):
    model = Autoencoder().cpu()
    print(model)
    params = list(model.parameters())
    print(len(params))
    print(params[0].size())

    distance = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in dataloader:
            pconn = data
            pconn = Variable(pconn[0]).cpu()
            output = model(pconn)
            loss = distance(output, pconn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))

    return(model)

def evaluate_autoencoder(model, test_dataloader):
    for ix, data in enumerate(test_dataloader):
        pconn = data
        pconn = Variable(pconn[0])
        output = model(pconn)

        input_pconn = reconstruct_pconn(pconn[0])
        output_pconn = reconstruct_pconn(output.data)
        make_compare_plot(input_pconn, output_pconn, ix)


def make_compare_plot(input_pconn, output_pconn, ix):
    fig, (ax1, ax2) = plt.subplots(1,2)
    im1 = ax1.imshow(input_pconn, cmap='hot')
    im2 = ax2.imshow(output_pconn, cmap='hot')
    plt.savefig("out_" + str(ix) + ".png")

def normalize(vec):
    return 2*(vec-vec.min())/(vec.max()-vec.min())-1

def reconstruct_pconn(pconn_vec):
    pconn_vec=normalize(pconn_vec)
    iu = np.triu_indices(NUM_PARCELS, 1)
    recon_pconn = np.zeros((NUM_PARCELS, NUM_PARCELS))
    np.fill_diagonal(recon_pconn,1)
    recon_pconn[iu] = pconn_vec
    recon_pconn = recon_pconn + recon_pconn.T - np.diag(np.diag(recon_pconn))
    return(recon_pconn)

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
        pconn_data[i] = normalize(pconn_tri)

    test_data = pconn_data[0:int(len(pconn_list) * SPLIT)-1]
    train_data = pconn_data[int(len(pconn_list) * SPLIT):-1]

    print(train_data.shape)
    print(test_data.shape)
    return (train_data, test_data)


def main():

    data_dir = '/Users/andersperrone/Projects/VAE/data/test_pconn_data'

    (train_data, test_data) = load_pconn_data(data_dir)

    train_tensor = torch.Tensor(train_data)
    train_dataset = TensorDataset(train_tensor)
    train_dataloader = DataLoader(train_dataset)
    model, train_autoencoder(train_dataloader)

    test_tensor = torch.Tensor(test_data)
    test_dataset = TensorDataset(test_tensor)
    test_dataloader = DataLoader(test_dataset)
    evaluate_autoencoder(model, test_dataloader)

if __name__ == "__main__":
    main()
