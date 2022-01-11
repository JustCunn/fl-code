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
num_selected = 20
num_models = 3

batch_size = 40
baseline_num = 100

epochs = 5
r_epochs = 20
embedding_dim = 64
hidden_dim = 64
vocab_size = 1901732

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataset():
    """
    with open('train.set', "rb") as f:
        x_train = cpickle.load(f)
        
    with open('test.set', "rb") as f:
        x_test = cpickle.load(f)
    """
    # We use an older PyTorch version with different dataloaders. They became deprecated in later versions.
    # We first loaded the PyTorch dataset object (that was already tokenised and a vocabulary created)
    # and saved it so it could be used with newer PyTorch versions

def clients_rand(train_len, num_clients):
    """
    Determines how much data each client has
    """
    
    client_temp = []
    sum = 0

    for i in range(num_clients - 1):
        temp = random.randint(10,100)
        sum += temp
        client_temp.append(temp)

    client_temp = np.array(client_temp)

    clients_dist= ((client_temp/sum)*train_len).astype(int)
    num  = train_len - clients_dist.sum()
    to_ret = list(clients_dist)
    to_ret.append(num)
    return to_ret


def split_data(data, labels, num_clients=num_clients, classes_per_client=classes_per_client, shuffle=True):
    '''
    Splits data among the clients
    '''
    
    #### constants #### 
    data_len = data.shape[0]
    n_labels = np.max(labels) + 1


    ### client distribution ####
    data_pc = clients_rand(len(data), num_clients)
    data_pc_pc = [np.maximum(1,nd // classes_per_client) for nd in data_pc]

    # sort for labels
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
            take = min(data_pc_pc[i], len(data_idxs[c]), budget)

            client_idxs += data_idxs[c][:take]
            data_idxs[c] = data_idxs[c][take:]

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
        data[i][0],data[i][1] = shuffle_list_data(data[i][0],data[i][1])
        
    return data

def shuffle_list_data(x, y):
    '''
    Shuffles Array
    '''
    idxs = list(range(len(x)))
    random.shuffle(idxs)
    return x[idxs],y[idxs]

class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = list(labels)

    def __getitem__(self, index):
        text, label = self.inputs[index], self.labels[index]

        return (text, label)

    def __len__(self):
        return self.inputs.shape[0]
    
    def to_list(self, inputs):
        l = []
        for i in inputs:
            f = list(i)
            l.append(f)
        return l

class CustomImageDataset(Dataset):
    '''
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''
    def __init__(self, inputs, labels):
        #assert inputs.shape[0] == labels.shape[0]
        self.inputs = inputs
        self.labels = list(labels)

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        return (label, img)

    def __len__(self):
        return len(self.inputs)

    def to_list(self, inputs):
        l = []
        for i in inputs:
            f = list(i)
            l.append(f)
        return l


def collate_batch(batch, label_pipeline):
    label_list, text_list, offsets = [], [], [0]
    for i in tqdm(batch):
         label_list.append(label_pipeline(i[0]))
         processed_text = i[1].numpy()
         text_list.append(processed_text)
         #offsets.append(processed_text.size(0))
    label_list = np.array(label_list)
    #offsets = np.array(offsets[:-1]).cumsum(dim=0)
    text_list = np.array(text_list)
    return label_list, text_list

def collate_batch_2(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

def collate_batch_3(batch):
    tokenizer = get_tokenizer('basic_english')
    label_pipeline = lambda x: int(x)
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = _text
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

def get_data_loaders(x_train, x_test, nclients,batch_size,classes_pc=classes_pc ,verbose=True ):

    label_pipeline = lambda x: int(x)

    y_train, x_train = collate_batch(x_train, label_pipeline)

    split = split_image_data(x_train, y_train, num_clients=num_clients, 
        classes_per_client=classes_per_client)

    split_temp = shuffle_list(split)
  
    client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y), 
                                                                batch_size=batch_size, shuffle=False, collate_fn=collate_batch_2) for x, y in split_temp]

    test_loader  = torch.utils.data.DataLoader(x_test, batch_size=100, shuffle=False, collate_fn=collate_batch_3) 

    return client_loaders, test_loader
  
import math

class MODEL(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(MODEL, self).__init__()
        self.embedding_size = embedding_dim
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Fully connected layer definition
        self.fc = nn.Linear(64, 10)
        
    def forward(self, x, offset):

        x = self.embedding(x, offset)
        x = x.unsqueeze(1)
    
        lstm_out, _ = self.lstm(x)

        lstm_out = lstm_out.view(lstm_out.size(0), -1)
        

        # The vector is passed through a fully connected layer
        out = self.fc(lstm_out)	
        # Activation function is applied
        output = F.log_softmax(out, dim=1)

        return output
