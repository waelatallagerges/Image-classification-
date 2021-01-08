import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import use

ap = argparse.ArgumentParser(description='Train.py')


ap.add_argument('data_dir', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
ap.add_argument('--arch', dest="arch", action="store", default="vgg13", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)



pa = ap.parse_args()
root = pa.data_dir
path = pa.save_dir
learning_rate = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hid1 = pa.hidden_units
device = pa.gpu
epochs = pa.epochs

def main():
    
    train_loader, v_loader, test_loader = use.load_data(root)
    model, optimizer, criterion = use.First_net(structure,dropout,hid1,learning_rate,device)
    use.trainig_network(model, optimizer, criterion, epochs, 20, train_loader, device)
    use.loading_model(structure,hid1,dropout,learning_rate)
    print("Done !")


if __name__== "__main__":
    main()