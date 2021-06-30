# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 19:24:30 2020

@author: Zhang Jingheng
"""

# hyper parameters
EPOCHS = 20
batch_size = 8
learning_rate = 0.001
classes = 2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import os
    
print(torch.cuda.is_available())##判断GPU是否可用
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
root="F:/isic/"                       #数据集本地存储位置

# -----------------ready the dataset--------------------------
def default_loader(path):
    im=Image.open(path).convert('RGB')
    return im.resize((224,224),Image.ANTIALIAS)   #使用PIL将图片调整为统一大小

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

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,drop_last=False)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True,drop_last=False)

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
                                                                                 #  128*128*3
#-----------------create the Net and training------------------------
def conv3x3(in_planes, out_planes, stride=1,groups=1, dilation=1): #封装一个3*3的卷积函数
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):   
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
   # 用于ResNet18和34的残差块，用的是2个3x3的卷积 
    expansion = 1
        
    def __init__(self, in_planes, planes, stride=1, downsample=None,groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d        
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
       
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=classes, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
           replace_stride_with_dilation = [False, False, False]                                                          
        self.groups = groups
        self.base_width = width_per_group                                                                                                            
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)  #128*128*16                           
        self.bn1 = nn.BatchNorm2d(self.inplanes)                                                     #64
        self.relu = nn.ReLU(inplace=True)                                                           # 32
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) ###########
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
    
        
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True):
    return  _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress)

model = resnet18()
model = model.cuda()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #定义优化器
loss_func = torch.nn.CrossEntropyLoss()           #定义损失函数        
        
# for updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
                
        
# train model
pre_epoch_total_step = len(train_loader)
current_lr = learning_rate
for epoch in range(EPOCHS):
    for i, (batch_x, batch_y) in enumerate(train_loader):
        x = batch_x.cuda()
        y = batch_y.cuda()

        # forward
        prediction = model(x)
        loss = loss_func(prediction, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            template = r"Epoch:{}/{}, step:{}/{}, Loss:{:.6f}"
            print(template.format(epoch+1, EPOCHS, i+1, pre_epoch_total_step, loss.item()))

    # decay learning rate
    if (epoch+1) % 20 == 0:
        current_lr = current_lr/2
        update_lr(optimizer, current_lr)

# test model
    model.eval()
    with torch.no_grad():
      total = 0
      correct = 0
      for x, y in test_loader:
        x = x.cuda()
        y = y.cuda()
        prediction = model(x)
        _, predic = torch.max(prediction.data, dim=1)
        total += y.size(0)
        correct += (predic == y).sum().item()

    print("Accuracy:{}%".format(100 * correct / total))       
        
 
#torch.save(model.state_dict(), "cifar10_resnet.ckpt")
        