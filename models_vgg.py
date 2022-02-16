'''
Copied from https://github.com/chengyangfu/pytorch-vgg-cifar10.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


#class VGG(nn.Module):
#
#    def __init__(self):
#        super(VGG, self).__init__()
#
#        #self.quant = QuantStub()
#        self.conv1 = nn.Conv2d(3,128,kernel_size=3,padding=1)
#        self.relu1=nn.ReLU(inplace=True)
#        self.conv2 = nn.Conv2d(128,128,kernel_size=3,padding=1)
#        self.relu2=nn.ReLU(inplace=True)
#        self.MaxPool2d1= nn.MaxPool2d(kernel_size=2,stride=2)
#        self.conv3 = nn.Conv2d(128,256,kernel_size=3,padding=1)
#        self.relu3=nn.ReLU(inplace=True)
#        self.conv4 = nn.Conv2d(256,256,kernel_size=3,padding=1)
#        self.relu4=nn.ReLU(inplace=True)
#        self.MaxPool2d2= nn.MaxPool2d(kernel_size=2,stride=2)
#        self.conv5 = nn.Conv2d(256,512,kernel_size=3,padding=1)
#        self.relu5=nn.ReLU(inplace=True)
#        self.conv6 = nn.Conv2d(512,512,kernel_size=3,padding=1)
#        self.relu6=nn.ReLU(inplace=True)
#        self.MaxPool2d3= nn.MaxPool2d(kernel_size=2,stride=2)
#        self.fc1 = nn.Linear(8192,1024)
#        self.relu7=nn.ReLU(inplace=True)
#        self.fc2 = nn.Linear(1024,10)
#        #self.dequant = DeQuantStub()
#
#    def forward(self, x):
#        #x = self.quant(x)
#        x = self.conv1(x)
#        x = self.relu1(x)
#        x = self.conv2(x)
#        x = self.relu2(x)
#        x = self.MaxPool2d1(x)
#        x = self.conv3(x)
#        x = self.relu3(x)
#        x = self.conv4(x)
#        x = self.relu4(x)
#        x = self.MaxPool2d2(x)
#        x = self.conv5(x)
#        x = self.relu5(x)
#        x = self.conv6(x)
#        x = self.relu6(x)
#        x = self.MaxPool2d3(x)
#        x = x.reshape(x.size(0), -1)
#        x = self.fc1(x)
#        x = self.relu7(x)
#        x = self.fc2(x)
#        #x = self.dequant(x)
#        return x




class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        # vgg8
        #self.classifier = nn.Sequential(
        #    nn.Linear(8192,1024),
        #    nn.ReLU(True),
        #    nn.Linear(1024,10)
        #)

            
        ## vgg11
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 10),
        ) 

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1 )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
    'F': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M']
}

def vgg8():
    """VGG 8-layer model(configuration "F")"""
    return VGG(make_layers(cfg['F']))
    #return VGG()

def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))



