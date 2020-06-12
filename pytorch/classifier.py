#!/usr/bin/env python

from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from captum.attr import IntegratedGradients

from autoencoder import Autoencoder


input_size = 242
l1_size = 121
l2_size = 27
l3_size = 7
output = 2

NUM_PARCELS = 352
num_epochs = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, l3_size)
        self.fc4 = nn.Linear(l3_size, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x)

def train_classifier(dataloader):
    classifier = Net().cpu()
    print(classifier)
    params = list(classifier.parameters())
    print(len(params))
    print(params[0].size())

    distance = nn.NLLLoss()

    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.0001, momentum=0.8)

    for epoch in range(num_epochs):
        for data, target in dataloader:
            data, target = Variable(data), Variable(target)
            out = classifier(data)
            target = target.long()
            loss = distance(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))

    return(classifier)


def normalize(vec):
    return 2*(vec-vec.min())/(vec.max()-vec.min())-1

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

    return(pconn_data)

def create_class_data():
    asd_dir = '/home/exacloud/lustre1/fnl_lab/projects/VAE/data/asd'
    asd_data = load_pconn_data(asd_dir)
    adhd_dir = '/home/exacloud/lustre1/fnl_lab/projects/VAE/data/adhd'
    adhd_data = load_pconn_data(adhd_dir)
    pconn_data = np.concatenate([asd_data, adhd_data], axis=0)

    labels = []
    for ix in range(0,asd_data.shape[0]):
        labels += [1]
    for ix in range(0,adhd_data.shape[0]):
        labels += [0]
    labels = np.array(labels)
    train_indices = np.random.choice(len(labels), int(0.7*len(labels)), replace=False)
    test_indices = list(set(range(len(labels))) - set(train_indices))
    train_pconns = pconn_data[train_indices]
    train_labels = labels[train_indices]
    test_pconns = pconn_data[test_indices]
    test_labels = labels[test_indices]
    return(train_pconns, train_labels, test_pconns, test_labels)

def evaluate_autoencoder(model, test_dataloader):
    for ix, data in enumerate(test_dataloader):
        pconn = data
        pconn = Variable(pconn[0]).to(dev)
        output, latent = model(pconn)
        output.to(dev)
        #print(latent)

        input_pconn_tensor = pconn[0]
        output_pconn_tensor = output.data
        input_pconn = reconstruct_pconn(input_pconn_tensor.cpu())
        output_pconn = reconstruct_pconn(output_pconn_tensor.cpu())
        make_compare_plot(input_pconn, output_pconn, ix)


def make_compare_plot(input_pconn, output_pconn, ix):
    fig, (ax1, ax2) = plt.subplots(1,2)
    im1 = ax1.imshow(input_pconn, cmap='hot')
    im2 = ax2.imshow(output_pconn, cmap='hot')
    plt.savefig("output/out_" + str(ix) + ".png")

def reconstruct_pconn(pconn_vec):
    pconn_vec=normalize(pconn_vec)
    #orig_pconn = np.zeros(61776)
    #orig_pconn[trunc_sorted_idx] = pconn_vec
    iu = np.triu_indices(NUM_PARCELS, 1)
    recon_pconn = np.zeros((NUM_PARCELS, NUM_PARCELS))
    np.fill_diagonal(recon_pconn,1)
    recon_pconn[iu] = pconn_vec
    recon_pconn = recon_pconn + recon_pconn.T - np.diag(np.diag(recon_pconn))
    return(recon_pconn)


def encode_data(model, input_tensor):
    return model.encoder(input_tensor)

def main():
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load('autoencoder_model.pt'))



    train_pconns, train_labels, test_pconns, test_labels = create_class_data()
    test_pconns_dataset = TensorDataset(torch.Tensor(test_pconns))
    test_pconns_dataloader = DataLoader(test_pconns)
    evaluate_autoencoder(autoencoder, test_pconns_dataloader)

    """
    # Encode and train classifier
    encoded_train_pconn_data = []
    for ix in range(0,train_pconns.shape[0]):
        y = encode_data(autoencoder, torch.Tensor(train_pconns[ix]))
        encoded_train_pconn_data += [y]
    encoded_train_tensor_data = torch.stack(encoded_train_pconn_data)
    encoded_test_pconn_data = []
    for ix in range(0,test_pconns.shape[0]):
        y = encode_data(autoencoder, torch.Tensor(test_pconns[ix]))
        encoded_test_pconn_data += [y]
    encoded_test_tensor_data = torch.stack(encoded_test_pconn_data)


    train_label_tensor_data = torch.Tensor(train_labels)
    train_class_dataset = TensorDataset(encoded_train_tensor_data, train_label_tensor_data)
    train_class_dataloader = DataLoader(train_class_dataset)

    classifier = train_classifier(train_class_dataloader)
    torch.save(classifier.state_dict(), '/home/exacloud/lustre1/fnl_lab/projects/VAE/code/pytorch/classifier_model.pt')

    out_probs = classifier(encoded_train_tensor_data).detach().numpy()
    out_classes = np.argmax(out_probs, axis=1)
    print("Train Accuracy:", sum(out_classes == train_labels) / len(train_labels))

    out_probs = classifier(encoded_test_tensor_data).detach().numpy()
    out_classes = np.argmax(out_probs, axis=1)
    print("Test Accuracy:", sum(out_classes == test_labels) / len(test_labels))

    ig = IntegratedGradients(classifier)
    attributions_ig, delta = ig.attribute(encoded_test_tensor_data, target=0, return_convergence_delta=True)
    attr = attributions_ig.detach().numpy()
    importances = np.mean(attr,axis=0)
    x_pos = (np.arange(len(importances)))
    plt.figure()
    plt.bar(x_pos, importances)
    plt.xticks(x_pos, x_pos)
    plt.xlabel("Encoded Features")
    plt.title("Feature Importance")
    plt.savefig('importance.png')
    """



if __name__ == "__main__":
    main()

