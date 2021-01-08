import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
from collections import OrderedDict
import argparse
arch = {"vgg13":25088,"alexnet":9216}

def load_data(where  = "./flowers" ):
    
    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    
    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])


    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    v_loader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)



    return train_loader , v_loader, test_loader

       
def First_net(structure='vgg13',dropout=0.5, hidden_layer1 = 120,lr = 0.001,device=gpu):
     if structure == 'vgg13':
        model = models.vgg13(pretrained=True)        
    elif structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("the used model is:".format(structure))
    
     model = models.vgg13(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('input', nn.Linear(structures[structure], hid1)),
            ('relu1', nn.ReLU()),
            ('hidlayer1', nn.Linear(hid1, 90)),
            ('relu2',nn.ReLU()),
            ('hidlayer2',nn.Linear(90,80)),
            ('relu3',nn.ReLU()),
            ('hidayer3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    model.classifier= classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate )
    if torch.cuda.is_available() and power = 'gpu':
      model.cuda()
    
    return model, optimizer ,criterion


def trainig_network(model, criterion, optimizer, epochs = 3, print_every=20, loader=train_loader, device='gpu'):
    steps = 0
    running_loss = 0
    for e in range(epochs):
            running_loss = 0
            for ii, (input, labels) in enumerate(train_loader):
                steps += 1

                input,labels = input.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(input)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()
                    vlost = 0
                    accuracy=0


                    for ii, (inputs2,labels2) in enumerate(v_loader):
                        optimizer.zero_grad()

                        inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                        model.to('cuda:0')
                        with torch.no_grad():    
                            outputs = model.forward(inputs2)
                            vlost = criterion(outputs,labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                    vlost = vlost / len(v_loader)
                    accuracy = accuracy /len(v_loader)



                    print("Epochs: {}/{}... ".format(e+1, epochs),
                          "Lossing: {:.4f}".format(running_loss/print_every),
                          "Validation_Lost {:.4f}".format(vlost),
                           "Accuracy: {:.4f}".format(accuracy))


                    running_loss = 0
                    
def save_checkpoint(path='checkpoint.pth',structure ='vgg13', hid1=120,dropout=0.5,learning_rate=0.001,epochs=20):
    
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    torch.save({'structure' :structure,
                'hidden_layer1':hid1,
                'dropout':dropout,
                'learning_rate':lr,
                'numberofepochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
    
def load_checkpoint(path='checkpoint.pth'):
    
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hid1 = checkpoint['hid1']
    dropout = checkpoint['dropout']
    learning_rate=checkpoint['learning_rate']

    model,_,_ = First_net(structure , dropout,hid1,learning_rate)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict']) 
    
def process_image(image_path):
    
    for i in image_path:
        path = str(i)
    img = Image.open(i) # Here we open the image

    make_img_good = transforms.Compose([ # Here as we did with the traini ng data we will define a set of
        # transfomations that we will apply to the PIL image
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_image = make_img_good(img)

    return tensor_image


def predict(image_path, model, topk=5,device='gpu'):
    
    if torch.cuda.is_available() and device='gpu':
        model.to('cuda:0')

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if device == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)    
                   
                    