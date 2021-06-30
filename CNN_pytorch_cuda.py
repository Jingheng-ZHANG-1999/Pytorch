# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 01:10:03 2020

@author: Zhang Jingheng
"""

import torch
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import os
use_gpu = torch.cuda.is_available()    ##判断GPU是否可用
print(use_gpu)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
root="F:/isic/"                       #数据集本地存储位置

# -----------------ready the dataset--------------------------
def default_loader(path):
    im=Image.open(path).convert('RGB')
    return im.resize((512,512),Image.ANTIALIAS)   #使用PIL将图片调整为统一大小

class MyDataset(Dataset):                           ##继承torch.utils.data.Dataset，并定义三个函数
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:                             #读入数据标签
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):                        #返回迭代器的索引范围
        return len(self.imgs)

train_data = MyDataset(txt=root+'train.txt', transform=transforms.ToTensor()) ##数据类型转换为tensor
test_data = MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=True)

def show_batch(imgs):                                ##定义显示图片的函数
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title("Batch from dataloader")


for i, (batch_x, batch_y) in enumerate(train_loader): ##显示前四批图片
    if(i<4):
        print(i, batch_x.size(),batch_y.size())
        show_batch(batch_x)
        plt.axis('off')
        plt.show()

#-----------------create the Net and training------------------------

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()               ##512*512*3
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),      ##512*512*32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)                 ##256*256*32
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),    ##256*256*64
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)               ##128*128*64
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),   ##128*128*64
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)              ##64*64*64
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, 3, 1, 1),   ##64*64*32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)              ##32*32*32
        )                        
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(32768, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)                                                 
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        res = conv4_out.view(conv4_out.size(0), -1)
       # print("-->{}".format(res.size()))
        out = self.dense(res)
        return out       

model = Net()
model = model.cuda()
print(model)

optimizer = torch.optim.Adam(model.parameters())  #定义优化器
loss_func = torch.nn.CrossEntropyLoss()           #定义损失函数

for epoch in range(2):                           #设定epoch次数
    print('epoch {}'.format(epoch + 1))
    
# training-------------------------------------------------------------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_loader:        
        batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda() #转换为variable类型
        out = model(batch_x)                          ##运行模型
        loss = loss_func(out, batch_y)                #计算loss
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()         #清空梯度
        loss.backward()               #计算出梯度
        optimizer.step()              #调用step方法，更新权重参数        
        loss = loss.cpu()             #将loss传回cpu
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_data)), train_acc / (len(train_data))))

# evaluation----------------------------------------------------------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
      with torch.no_grad():  
        batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
        loss = loss.cpu()        
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_data)), eval_acc / (len(test_data))))






