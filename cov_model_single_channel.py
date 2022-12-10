import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

import torch.nn as nn 
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix

import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, fftfreq

class CNNModel(nn.Module):
  def __init__(self):
    super(CNNModel, self).__init__()
    
    self.cnn1 = nn.Conv1d(in_channels=1, out_channels=21, kernel_size=3, stride=1, padding=0)
    self.relu1 = nn.ReLU() 
    self.cnn2 = nn.Conv1d(in_channels=21, out_channels=23, kernel_size=3, stride=1, padding=0)
    self.relu2 = nn.ReLU()  
  

    self.fc1 = torch.nn.Linear(1518, 128)
    self.fc2 = torch.nn.Linear(128, 4)

  def forward(self,x):

    out = self.cnn1(x)
    out = self.relu1(out)
    out = self.cnn2(out)      
    out = self.relu2(out)

    out = out.view(out.size(0), -1)


    out = self.fc1(out)
    out = self.fc2(out)      
    #out = self.fc3(out)
    #out = self.fc4(out)
    #out = self.fc5(out)
    
    return out




def create_thing(channel):
        
    tedata = np.load('x_test.npy')
    trdata = np.load('x_train.npy')

    def create_power_spectra(data):
        new_data = []
        for ex in data:
            ex = ex[channel, :]
        
            new_data.append([])
           
            yf = fft(ex)
            new_data[-1].append(np.abs(yf)[:70])
            
        new_data = np.array(new_data)
        return new_data

    train = create_power_spectra(trdata)
   
    test = create_power_spectra(tedata)
  


    np.save(f'x_train_power{channel}', train)
    np.save(f'x_test_power{channel}', test)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNNModel()
model.to(device)

error = torch.nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 10


for channel in range(19):
    
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []

    create_thing(channel)

    batch_size = 1
    train_X = np.load(f'x_train_power{channel}.npy')
    train_Y = np.load('y_train.npy')
    test_X = np.load(f'x_test_power{channel}.npy')
    test_Y = np.load('y_test.npy')
    FULL_X = np.concatenate((train_X,test_X))    
    FULL_Y = np.concatenate((train_Y, test_Y))
    # Redifine Train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(FULL_X, FULL_Y, test_size=0.33, random_state=69)
    print(X_train.shape)
    print(Y_train.shape)
    print(Y_train.shape)
    print(Y_test.shape)


    labels_train = F.one_hot(torch.Tensor(Y_train).type(torch.long)).type(torch.float)
    train_set = TensorDataset(torch.Tensor(X_train).type(torch.float), labels_train)
    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)


    labels_test = F.one_hot(torch.Tensor(Y_test).type(torch.long)).type(torch.float)
    test_set = TensorDataset(torch.Tensor(X_test).type(torch.float), labels_test)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNNModel()
    model.to(device)

    error = torch.nn.CrossEntropyLoss()

    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 10
    for epoch in range(num_epochs):
        #model.train()
        running_loss = 0
        correct = 0
        for i, (x_data, labels) in enumerate(train_loader):
            x_data, labels = x_data.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(x_data)

            loss = error(outputs, labels)
            
            loss.backward()
            
            optimizer.step()
            
            correct += (torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).float().sum()
            running_loss += loss.item()

        acc = correct / len(train_set)
        print('(%d) train loss= %.3f; train accuracy = %.1f%%' % (epoch, loss, 100 * acc))

    correct = 0
    for i, (x_data, labels) in enumerate(test_loader):
            x_data, labels = x_data.to(device), labels.to(device)
            outputs = model(x_data)
        
            correct += (torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).float().sum()

    print(correct, len(test_set))
    acc = correct / len(test_set)
    print('(%d) test loss= %.3f; test accuracy = %.1f%%' % (100, loss, 100 * acc))


    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in test_loader:
            output = model(inputs) # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            labels = np.argmax(labels, axis=1)
            y_true.extend(labels) # Save Truth

    # constant for classes
    y_pred = torch.Tensor(y_pred).type(torch.float) 
    y_true = torch.Tensor(y_true).type(torch.float)
    normalized_metric = MulticlassConfusionMatrix(num_classes = 4, normalize='true')
    raw_metric = MulticlassConfusionMatrix(num_classes = 4)
    metric = normalized_metric(y_pred,y_true)
    raw_metric = raw_metric(y_pred,y_true)
    classes = ['Normal', 'Complex', 'Electro', 'Video']
    df_cm = pd.DataFrame(metric, index = [i for i in classes], columns = [i for i in classes])
    print(metric)
    print(raw_metric)
    fig, ax = plt.subplots(figsize=(15,10))         # Sample figsize in inches
    sn.heatmap(metric, annot=True, ax = ax)
    plt.savefig(f'heatmap_channel{channel}.png')
