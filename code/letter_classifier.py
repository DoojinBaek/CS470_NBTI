import os
import torch
import numpy as np
import pandas as pd
import torch.nn.Functional as F


########## helper functions ##########
def cal_acc(pred, label):
    '''
    pred: batch_size x 26 with probabilities
    label: batch_size x 26 in one-hot format
    '''
    pred_word = torch.argmax(pred, axis=1)
    label_word = torch.argmax(label, axis=1)
    correct_pred_counts = torch.sum(pred_word == pred_word)
    
    acc = correct_pred_counts.item() / len(pred) # just float not torch
    
    return acc

def calculate_val_loss_and_acc(model, loss_fn, dataloader):
    loss_buffer = []
    acc_buffer = []
    with torch.no_grad():
        for idx, (data, label) in enumerate(dataloader):
            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            loss = loss_fn(pred, label)
            acc = cal_acc(pred, label)
            loss_buffer.append(loss.item())
            acc_buffer.append(acc)
    
    return np.mean(loss_buffer), np.mean(acc_buffer)



########## Model ##########
class Classifier(torch.nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(20, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(32*9*9, 256, bias=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 26)
        )
        self.dropout = torch.nn.Dropout(0.3)
    
    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        out = self.layer4(x.reshape(x.shape[0], -1)) # input: batch_size x all_features

        return out # batch_size x 26



########## Training Code ##########
def train(model, learning_rate, train_dataloader, valid_dataloader, device):
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    val_loss_min = np.inf
    val_acc_max = -np.inf

    valid_loss_buffer = []
    train_loss_buffer = []
    valid_acc_buffer = []
    train_acc_buffer = []

    # validation loss and accuracy
    val_loss, val_acc = calculate_val_loss_and_acc(model, loss_fn, valid_dataloader) # initial model
    valid_loss_buffer.append(val_loss)
    valid_acc_buffer.append(val_acc)

    for epoch in range(num_epochs):
        for idx, (data, label) in enumerate(train_dataloader):

            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            loss = loss_fn(pred, label)
            acc = cal_acc(pred, label)
            train_loss_buffer.append(loss)
            train_acc_buffer.append(acc)

            # update the model
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            val_loss, val_acc = calculate_val_loss_and_acc(model, loss_fn, valid_dataloader)
            valid_loss_buffer.append(val_loss)
            valid_acc_buffer.append(val_acc)

            if val_loss < val_loss_min:
                print('new minimum validation loss:', val_loss)
                val_loss_min = val_loss
                torch.save(model.state_dict(), "./checkpoints/min_val_loss_checkpoint.pt")

            if val_acc > val_acc_max:
                print('new maximum validation accuracy:', val_acc)
                val_acc_max = val_acc
                torch.save(model.state_dict(), "./checkpoints/max_val_acc_chaekpoint.pt")

            if (epoch+1)%10 == 0:
                torch.save(model.state_dict(), "./checkpoints/epoch_{}.pt".format(epoch+1))


if __name__ == '__main__':
    
    device = 'cuda'
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001

    ########## To Do ##########
    '''

    load data using torch.
        x_train, y_train
        x_val, y_val

    '''
    ###########################
    train_dataloader = None
    valid_dataloader = None

    model = Classifier().to(device)
    
    train(model, learning_rate, train_dataloader, valid_dataloader, device)
