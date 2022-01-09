# -*- coding: utf-8 -*-
import os
import random
from tqdm import tqdm

import random
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms 
from torchvision.transforms import Compose 
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark=True

classes_per_client = 2

num_clients = 20
num_selected = 6
num_models = 15
stale_prob = 2
stale_hist = 10

batch_size = 40
baseline_num = 100

epochs = 5
r_epochs = 20

def get_datasets():
    training_data = torchvision.datasets.CIFAR10('./data', train=True, download=True)
    testing_data = torchvision.datasets.CIFAR10('./data', train=False, download=False) 

    x_train, y_train = training_data.data.transpose((0,3,1,2)), np.array(training_data.targets)
    x_test, y_test = testing_data.data.transpose((0,3,1,2)), np.array(testing_data.targets)
  
    return x_train, y_train, x_test, y_test
  
def clients_rand(train_len, num_clients):
    """
    Determines how much data each client has
    """
    
    client_temp = []
    sum = 0

    for i in range(num_clients - 1):
        temp = random.randint(1,100)
        sum += temp
        client_temp.append(temp)

    client_temp = np.array(client_temp)

    clients_dist = ((client_temp/sum)*train_len).astype(int) 
    remain_num = train_len - clients_dist.sum() # Remaining client size
    client_sizes = list(clients_dist)
    client_sizes.append(remain_num)
    return to_ret


def split_data(data, labels, num_clients=num_clients, classes_per_client=classes_per_client, shuffle=True):
    '''
    Splits data among the clients
    '''
     
    data_len = data.shape[0]
    n_labels = np.max(labels) + 1


    ### client distribution ####
    data_pc = clients_rand(len(data), num_clients)
    data_pc_pc = [np.maximum(1,nd // classes_per_client) for nd in data_pc] # Data per client, per client

    # label sorting
    data_idxs = [[] for i in range(n_labels)]
    
    for i, label in enumerate(labels):
        data_idxs[label] += [i]
    if shuffle:
        for idxs in data_idxs:
            np.random.shuffle(idxs)
    
    # split data among clients
    client_split = []
    
    c = 0
    
    for i in range(num_clients):
        client_idxs = []

        budget = data_pc[i]
        c = np.random.randint(n_labels)
        while budget > 0:
            amount_to_get = min(data_pc_pc[i], len(data_idxs[c]), budget)

            client_idxs += data_idxs[c][:amount_to_get]
            data_idxs[c] = data_idxs[c][amount_to_get:]

            budget -= take
            c = (c + 1) % n_labels
        client_split += [(data[client_idxs], labels[client_idxs])]

    client_split = np.array(client_split)

    return client_split


def shuffle_list(data):
    '''
    This function returns the shuffled data
    '''
    for i in range(len(data)):
        temp_len= len(data[i][0])
        idx = [i for i in range(temp_len)]
        random.shuffle(idx)
        data[i][0],data[i][1] = shuffle_data(data[i][0],data[i][1])
        
    return data

def shuffle_data(x, y):
    '''
    Shuffles Array
    '''
    idxs = list(range(len(x)))
    random.shuffle(idxs)
    return x[idxs],y[idxs]

class CustomDataset(Dataset):
    def __init__(self, inputs, labels, transforms=None):
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms 

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]
    

def get_data_loaders(num_clients,batch_size,classes_per_client=classes_per_client):

    x_train, y_train, x_test, y_test = get_datasets()

    transforms_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transforms_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    split = split_data(x_train, y_train, num_clients=num_clients, 
        classes_per_client=classes_per_client)

    final_splits = shuffle_list(split)
  
    client_loaders = [torch.utils.data.DataLoader(CustomDataset(x, y, transforms_train), batch_size=batch_size, shuffle=True) for x, y in final_splits]

    test_loader  = torch.utils.data.DataLoader(CustomDataset(x_test, y_test, transforms_test), batch_size=100, shuffle=False) 

    return client_loaders, test_loader

def baseline_dataloader(num):
    '''
    Loads Baseline Loader
    num: Baseline data size
    Returns:
        loader: Baseline Loader
    '''
    x_train, y_train, x_temp, y_temp = get_datasets()
    x , y = shuffle_list_data(x_train, y_train)

    x, y = x[:num], y[:num]
    transform_baseline = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    loader = torch.utils.data.DataLoader(CustomDataset(x, y, transform_baseline), batch_size=16, shuffle=True)

    return loader

vgg19 = [64, 64, 'POOL', 128, 128, 'POOL', 256, 256, 256, 256, 'POOL', 512, 512, 512, 512, 'POOL', 512, 512, 512, 512, 'POOL']

class MODEL(nn.Module):
    def __init__(self, vgg):
        super(MODEL, self).__init__()
        self.features = self._make_layers(vgg)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        output = F.log_softmax(out, dim=1)
        return output

    def _make_layers(self, vgg):
        layers = []
        in_channels = 3
        for x in vgg:
            if x == 'POOL':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def update(client_model, optimizer, train_data_loader, epochs=epochs):
    """
    This function updates/trains client model on client data
    """
    
    client_model.train()
    
    for epoch in range(epochs):
        for x, y in train_data_loader:
            x, y = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(x)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
    return loss.item()

def test_func(global_model, test_loader):
    """
    Test Function
    """
    
    global_model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = data.cuda(), target.cuda()
            output = global_model(x)
            loss += F.nll_loss(output, y, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return loss, acc

def load_model(client_model, model_to_load):
    '''
    Loads the given model (model) for the client_model
    '''
    client_model.load_state_dict(model_to_load.state_dict())

def aggregate(model, clients, lengths):
    """
    Aggregation. No weighted mean as there's only one client
    """ 
    global_state_dict = model.state_dict()
    
    for i in global_state_dict.keys():
        client_dict = clients.state_dict()[i].float()
        global_state_dict[i] = torch.stack(client_dict, 0)
        
    model.load_state_dict(global_state_dict)

    clients.load_state_dict(model.state_dict())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

global_models =  [MODEL(vgg19).to(device) for _ in range(num_models)]
client_models = [MODEL(vgg19).to(device) for _ in range(num_selected)]

for i in client_models:
    glob_model = global_models[0].state_dict()
    i.load_state_dict(glob_model) 

opt = [optim.SGD(i.parameters(), lr=0.1) for i in client_models]

baseline_loader = baseline_dataloader(baseline_num)

train_loader, test_loader = get_data_loaders(classes_per_client = classes_per_client, num_clients = num_clients, batch_size = batch_size)

loss_train = [[] for _ in range(num_models)]
loss_test = [[] for _ in range(num_models)]
acc_test = [[] for _ in range(num_models)]
loss_retrain_list=[]

model_dicts = [[] for _ in range(num_models)]

client_index = np.random.permutation(num_clients)[:num_selected]
updates = 0

while True:
    for r in range(num_selected):
        # Iterate over clients

        client_lengths = [len(train_loader[i]) for i in client_index]

        for i in tqdm(range(num_models)):
            # Iterate over modelws
            model_dicts[i].append(global_models[i].state_dict()) # Save the model before training and changing it. We can use these for staleness
            
            if random.randint(1, stale_prob) == 2:
                try:
                  t = random.randint(1, stale_hist)
                  global_models[i].load_state_dict(model_dicts[i][(r-t)])
                except IndexError:
                  global_models[i].load_state_dict(model_dicts[i][(r-1)])
                
            loss = 0
            loss_retrain = 0
            
            updates += 25
            
            load_model(client_models[r], global_models[i])
            loss += update(client_models[r], opt[r], train_loader[client_index[r]], epochs)
            loss_retrain += update(client_models[r], opt[r], baseline_loader, r_epochs)
            loss_retrain_list.append(loss_retrain)
            loss_train[i].append(loss)
            aggregate(global_models[i], client_models[r],client_lengths[r])

            test_loss, acc = test_func(global_models[i], test_loader)
            loss_test[i].append(test_loss)
            acc_test[i].append(acc)
            print(f'Model {i+1}')
            print(f'Client {r} | Train Loss: {loss_retrain_list[-1]} | Test Accuracy: {acc} | Test Loss: {test_loss} | Update: {updates}')
