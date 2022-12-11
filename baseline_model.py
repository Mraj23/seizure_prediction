import torch.nn.functional as F
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassConfusionMatrix

class BaselineModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(70, 1) # Input to 1 dimension
        self.fc2 = torch.nn.Linear(19, 4) # Number of class labels

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.transpose(x,-1,-2)
        x = self.fc2(x)
        return x

train_X = np.load('x_train_power.npy')
train_Y = np.load('y_train.npy')
test_X = np.load('x_test_power.npy')
test_Y = np.load('y_test.npy')
FULL_X = np.concatenate((train_X,test_X))    
FULL_Y = np.concatenate((train_Y, test_Y))
# Redifine Train and test set
X_train, X_test, Y_train, Y_test = train_test_split(FULL_X, FULL_Y, test_size=0.33, random_state=69)


def train_model():
    model = BaselineModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Convert Y labels to one hot vectors
    labels = F.one_hot(torch.Tensor(Y_train).type(torch.long)).type(torch.float)
    train_set = TensorDataset(torch.Tensor(X_train).type(torch.float), labels)
    train_loader = DataLoader(dataset=train_set, shuffle=True)

    for epoch in range(10):
        running_loss = 0
        correct = 0
        for i, data in enumerate(train_loader):
            x_batch, y_batch = data
            optimizer.zero_grad()
            yhat = model(x_batch) 
            loss = criterion(yhat[0], y_batch.type(torch.float))
            loss.backward()
            optimizer.step()
            correct += (torch.argmax(yhat[0], dim=1) == torch.argmax(y_batch, dim=1)).float().sum()
            running_loss += loss.item()
        acc = correct / len(train_set)
        print('(%d) loss= %.3f; accuracy = %.1f%%' % (epoch, loss, 100 * acc))
    
    torch.save(model, 'baseline.pt')
    return model

def test_model(model):

    labels = F.one_hot(torch.Tensor(Y_test).type(torch.long)).type(torch.float)
    test_set = TensorDataset(torch.Tensor(X_test).type(torch.float), labels)
    test_loader = DataLoader(dataset=test_set, shuffle=False)
    correct = 0
    pred_label = []
    for i, data in enumerate(test_loader):
        x_batch, y_batch = data
        yhat = model(x_batch) 
        correct += (torch.argmax(yhat[0], dim=1) == torch.argmax(y_batch, dim=1)).float().sum()
        pred_label.append(torch.argmax(yhat[0], dim=1))
      
        
    print(correct, len(test_set), correct / len(test_set))
    pred_label = torch.FloatTensor(pred_label)
    return pred_label




if __name__ == '__main__':
    model = train_model()
    pred = test_model(model)
    
    target = torch.Tensor(Y_test).type(torch.float)
    print(pred.shape)
    print(target.shape)
    normalized_metric = MulticlassConfusionMatrix(num_classes = 4, normalize='true')
    raw_metric = MulticlassConfusionMatrix(num_classes = 4)
    metric = normalized_metric(pred,target)
    raw_metric = raw_metric(pred,target)
    classes = ['Normal', 'Complex', 'Electro', 'Video']
    df_cm = pd.DataFrame(metric, index = [i for i in classes], columns = [i for i in classes])
    print(raw_metric)
    sn.heatmap(df_cm, annot=True)
    plt.show()
    
