#%%
import os
import time
import torch
import numpy as np


import os
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets

from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torchsummary import summary

import torchvision
import matplotlib.pyplot as plt

import sys

exp_idx = sys.argv[1]

if len(sys.argv) != 2:
    exp_idx = 999

########## helper functions ##########
def invert(image):
    return transforms.functional.invert(image)
def cal_acc(pred, label):
    '''
    pred: batch_size x 52 with probabilities
    label: batch_size x 52 in one-hot format
    '''
#     print(pred.shape)
#     print(label.shape)
    pred_word = torch.argmax(pred, axis=1)
    label_word = label
    correct_pred_counts = torch.sum(pred_word == label_word)
    
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

data_dir = "./"

def create_datasets(batch_size):

    train_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.RandomRotation([-30, 30], fill=255),
    transforms.RandomPerspective(distortion_scale=0.65, p=0.7, fill=255),
    transforms.Lambda(invert),
    transforms.ToTensor()
])

    # choose the training and test datasets
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'data/letter_classifier'), train_transform)

    # trainning set 중 validation 데이터로 사용할 비율
    valid_size = 0.3

    # validation으로 사용할 trainning indices를 얻는다.
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # trainning, validation batch를 얻기 위한 sampler정의
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=4)

    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=4)

    return train_data, train_loader, valid_loader

def imshow(input, title):
    # torch.Tensor를 numpy 객체로 변환
    input = input.numpy().transpose((1, 2, 0))
    input = np.clip(input, 0, 1)

    # 이미지 출력
    plt.imshow(input)
    plt.title(title)
    plt.show()

class Classifier_CNN(torch.nn.Module):
    def __init__(self):
        super(Classifier_CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=4, stride=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.SiLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=4),
            torch.nn.BatchNorm2d(32),
            torch.nn.SiLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.SiLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3),
            torch.nn.BatchNorm2d(128),
            torch.nn.SiLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.SiLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.SiLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.SiLU()
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.Dropout1d(0.2),
            torch.nn.BatchNorm1d(1024),
            torch.nn.SiLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 52)
        )
    
    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.reshape(x.shape[0], -1)
        out = self.layer8(x)

        return out # batch_size x 52


########## Training Code ##########
def train(model, learning_rate, train_dataloader, valid_dataloader, device):
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, patience = 20, 
                                                           threshold=0.00005)
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
    
    print(summary(model, (1, 600, 600)))
    
    for epoch in range(num_epochs):
        
        train_loss_buffer_tmp = []
        train_acc_buffer_tmp = []
        print('Epoch: {}'.format(epoch))
        start = time.time()
        for idx, (data, label) in enumerate(train_dataloader):

            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            loss = loss_fn(pred, label)
            acc = cal_acc(pred, label)
            train_loss_buffer_tmp.append(loss.item())
            train_acc_buffer_tmp.append(acc)

            # update the model
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        val_loss, val_acc = calculate_val_loss_and_acc(model, loss_fn, valid_dataloader)
        valid_loss_buffer.append(val_loss)
        valid_acc_buffer.append(val_acc)
        train_loss_buffer.append(np.mean(train_loss_buffer_tmp))
        train_acc_buffer.append(np.mean(train_acc_buffer_tmp))

        if val_loss < val_loss_min:
            print('new minimum validation loss:', val_loss)
            val_loss_min = val_loss
            torch.save(model.state_dict(), "./checkpoints/{}/min_val_loss_checkpoint.pt".format(exp_idx))

        if val_acc > val_acc_max:
            print('new maximum validation accuracy:', val_acc)
            val_acc_max = val_acc
            torch.save(model.state_dict(), "./checkpoints/{}/max_val_acc_chaekpoint.pt".format(exp_idx))

        if (epoch+1)%10 == 0:
            torch.save(model.state_dict(), "./checkpoints/{}/epoch_{}.pt".format(exp_idx, epoch+1))
        if (epoch+1)%30 == 0:
            logging(exp_idx, valid_loss_buffer, train_loss_buffer, valid_acc_buffer, train_acc_buffer)
            
        done = time.time()
        print('Elapsed Time: {:.5f}'.format((done-start)/60))
        logging(exp_idx, valid_loss_buffer, train_loss_buffer, valid_acc_buffer, train_acc_buffer)
    
    
def logging(exp_idx, valid_loss_buffer, train_loss_buffer, valid_acc_buffer, train_acc_buffer):
    np.save('./logs/{}/valid_loss.npy'.format(exp_idx), valid_loss_buffer)
    np.save('./logs/{}/train_loss.npy'.format(exp_idx), train_loss_buffer)
    np.save('./logs/{}/valid_acc.npy'.format(exp_idx), valid_acc_buffer)
    np.save('./logs/{}/train_acc.npy'.format(exp_idx), train_acc_buffer)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_buffer, label='train', color='blue')
    plt.plot(valid_loss_buffer, label='valid', color='red')
    plt.legend()
    plt.xlabel('Epochs')
    plt.title('Cross Entropy Loss')
    plt.savefig('./logs/{}/loss.png'.format(exp_idx))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_buffer, label='train', color='blue')
    plt.plot(valid_acc_buffer, label='valid', color='red')
    plt.legend()
    plt.xlabel('Epochs')
    plt.title('Accuracy')
    plt.savefig('./logs/{}/acc.png'.format(exp_idx))
    plt.close()

if __name__ == '__main__':
    
    device = 'cuda'
    batch_size = 64
    num_epochs = 5000
    learning_rate = 0.001
    os.makedirs('./logs/{}/'.format(exp_idx), exist_ok=True)
    os.makedirs('./checkpoints/{}/'.format(exp_idx), exist_ok=True)

    train_data, train_loader, valid_loader = create_datasets(batch_size=batch_size)
    
    # 학습 데이터를 배치 단위로 불러오기
    iterator = iter(train_loader)
    class_names = train_data.classes

    # 현재 배치를 이용해 격자 형태의 이미지를 만들어 시각화
    inputs, classes = next(iterator)
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])
    
    # 학습 데이터를 배치 단위로 불러오기
    iterator = iter(valid_loader)

    # 현재 배치를 이용해 격자 형태의 이미지를 만들어 시각화
    inputs, classes = next(iterator)
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])


    print('Number of training dataset:', len(train_data))
    print('Number of validation dataset', len(valid_loader))

    class_names = train_data.classes
    print('Number of classes:', len(class_names))
    
    model = Classifier_CNN().to(device)
    
    print("Model Initialized")
    train(model, learning_rate, train_loader, valid_loader, device)

# %%
