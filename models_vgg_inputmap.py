'''
Copied from https://github.com/chengyangfu/pytorch-vgg-cifar10.git
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}

# A: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
class VGG11_bn(nn.Module):
    def __init__(self):
        super(VGG11_bn, self).__init__()

        self.conv2d1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2d2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2d3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv2d4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2d5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv2d6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2d7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv2d8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.dp1 = nn.Dropout()
        self.fc1 = nn.Linear(512, 512)
        self.dp2 = nn.Dropout()
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

        self.scale = [
            torch.ones(3,32,32).cuda(), 
            torch.ones(64,16,16).cuda(), 
            torch.ones(128,8,8).cuda(), 
            torch.ones(256,8,8).cuda(), 
            torch.ones(256,4,4).cuda(), 
            torch.ones(512,4,4).cuda(), 
            torch.ones(512,2,2).cuda(), 
            torch.ones(512,2,2).cuda(), 
            torch.ones(512).cuda(), 
            torch.ones(512).cuda(), 
            torch.ones(512).cuda()]
        for k in self.scale:
            k.requires_grad = True

    def forward(self, x):

        # x = self.features(x)
        x = F.relu(self.bn1(self.conv2d1(x * self.scale[0])))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2d2(x * self.scale[1])))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv2d3(x * self.scale[2])))
        x = F.relu(self.bn4(self.conv2d4(x * self.scale[3])))
        x = self.pool3(x)
        x = F.relu(self.bn5(self.conv2d5(x * self.scale[4])))
        x = F.relu(self.bn6(self.conv2d6(x * self.scale[5])))
        x = self.pool4(x)
        x = F.relu(self.bn7(self.conv2d7(x * self.scale[6])))
        x = F.relu(self.bn8(self.conv2d8(x * self.scale[7])))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.dp1(x * self.scale[8])))
        x = F.relu(self.fc2(self.dp2(x * self.scale[9])))
        y = self.fc3(x * self.scale[10])
        return y

    def load_para(net):
        for (num, p) in enumerate(net.parameters()):
            for (num_copy, p_copy) in enumerate(self.parameters()):
                if num_copy == num:
                    p_copy.data = p.data


# D: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
class VGG16_bn(nn.Module):
    def __init__(self):
        super(VGG16_bn, self).__init__()

        self.conv2d1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2d2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2d3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv2d4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2d5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv2d6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv2d7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2d8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv2d9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv2d10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2d11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv2d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv2d13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.dp1 = nn.Dropout()
        self.fc1 = nn.Linear(512, 512)
        self.dp2 = nn.Dropout()
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

        self.scale = [
            torch.ones(3,32,32).cuda(), 
            torch.ones(64,32,32).cuda(),
            torch.ones(64,16,16).cuda(), 
            torch.ones(128,16,16).cuda(), 
            torch.ones(128,8,8).cuda(), 
            torch.ones(256,8,8).cuda(), 
            torch.ones(256,8,8).cuda(), 
            torch.ones(256,4,4).cuda(), 
            torch.ones(512,4,4).cuda(), 
            torch.ones(512,4,4).cuda(), 
            torch.ones(512,2,2).cuda(), 
            torch.ones(512,2,2).cuda(), 
            torch.ones(512,2,2).cuda(), 
            torch.ones(512).cuda(), 
            torch.ones(512).cuda(), 
            torch.ones(512).cuda()]
        for k in self.scale:
            k.requires_grad = True

    def forward(self, x):
        x = F.relu(self.bn1(self.conv2d1(x * self.scale[0])))
        x = F.relu(self.bn2(self.conv2d2(x * self.scale[1])))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv2d3(x * self.scale[2])))
        x = F.relu(self.bn4(self.conv2d4(x * self.scale[3])))
        x = self.pool2(x)

        x = F.relu(self.bn5(self.conv2d5(x * self.scale[4])))
        x = F.relu(self.bn6(self.conv2d6(x * self.scale[5])))
        x = F.relu(self.bn7(self.conv2d7(x * self.scale[6])))
        x = self.pool3(x)

        x = F.relu(self.bn8(self.conv2d8(x * self.scale[7])))
        x = F.relu(self.bn9(self.conv2d9(x * self.scale[8])))
        x = F.relu(self.bn10(self.conv2d10(x * self.scale[9])))
        x = self.pool4(x)

        x = F.relu(self.bn11(self.conv2d11(x * self.scale[10])))
        x = F.relu(self.bn12(self.conv2d12(x * self.scale[11])))
        x = F.relu(self.bn13(self.conv2d13(x * self.scale[12])))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.dp1(x * self.scale[13])))
        x = F.relu(self.fc2(self.dp2(x * self.scale[14])))
        y = self.fc3(x * self.scale[15])
        return y


