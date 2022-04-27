import numpy as np
import cv2
import torch
import torchvision
import os
import torch
# import torch.nn as nn
# import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import torch.optim as optim
from matplotlib import pyplot as plt

X_train= np.load('train_X64_2D.npy')
targets_train= np.load('train_Y64_2D.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(X_train.shape)
X_train= X_train.reshape(52155,3,64,64)
# train_x,train_y=data_trasform("train"))
train_x = torch.from_numpy(np.array(X_train)).float()
train_y = torch.from_numpy(np.array(targets_train)).long()
# test_x = torch.from_numpy(np.array(X_test)).float()
# test_y = torch.from_numpy(np.array(targets_test)).long()
# train_x = torch.from_numpy(train_x).float()
batch_size = 32 #We pick beforehand a batch_size that we will use for the training


# Pytorch train and test sets
train = torch.utils.data.TensorDataset(train_x,train_y)
# test = torch.utils.data.TensorDataset(test_x,test_y)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
# test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
num_classes = 5

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.conv5 = nn.Conv2d(128, 128, 5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.pool = nn.MaxPool2d(1,1)
        self.batch=nn.BatchNorm1d(256)
        self.drop=nn.Dropout(p=0.15) 
        
    def forward(self, x):
#         print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
#         print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
#         print(x.shape)
        x = self.pool(F.relu(self.conv3(x)))
#         print(x.shape)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

#         print(x.shape)
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.batch(self.fc1(x)))
        x = self.drop(x)
        x = self.fc2(x)
        
        

        return x

#Definition of hyperparameters
# n_iters = 10000
# num_epochs = n_iters / (len(train_x) / batch_size)
# num_epochs = int(num_epochs)
# print(num_epochs)

# Create CNN
model = CNNModel()
#model.cuda()
print(model)
## Loss and optimizer
learning_rate = 1e-4 #I picked this because it seems to be the most used by experts
load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate) #Adam seems to be the most popular for deep learning
train_losses=[]
epochs=[]
num_epochs=30
for epoch in range(num_epochs): 
    loss_ep = 0
    model.train()
    epochs.append(epoch)
    for batch_idx, (data, targets) in enumerate(train_loader):
#         print(batch_idx,"/",epoch,"-------->")
        data = data
        targets = targets
        ## Forward Pass
        optimizer.zero_grad()
        scores = model(data)
        loss = criterion(scores,targets)
        loss.backward()
        optimizer.step()
        loss_ep += loss.item()
    train_losses.append(loss_ep/len(train_loader))
    print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_loader)}")

# visualization loss 
iteration_list=[i for i in range(num_epochs)]
plt.plot(epochs,train_losses)
plt.xlabel("Number of epoch")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of epoch")
plt.savefig('loss_plot_'+str(num_epochs)+'.png')
torch.save(model.state_dict(), "deep2d_30.pth")