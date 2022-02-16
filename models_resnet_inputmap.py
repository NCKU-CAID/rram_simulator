import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet18_inputmap(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_inputmap, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.shortcut1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn1 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.shortcut2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn2 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        self.shortcut3 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        self.shortcut_bn3 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(128)
        self.shortcut4 = nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn4 = nn.BatchNorm2d(128)

        self.conv10 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(256)
        self.shortcut5 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        self.shortcut_bn5 = nn.BatchNorm2d(256)

        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(256)
        self.shortcut6 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn6 = nn.BatchNorm2d(256)

        self.conv14 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        self.shortcut7 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.shortcut_bn7 = nn.BatchNorm2d(512)

        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(512)
        self.conv17 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(512)
        self.shortcut8 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn8 = nn.BatchNorm2d(512)

        self.linear = nn.Linear(512, num_classes)

        self.scale = [
            torch.ones(3,32,32).cuda(),  #0
            torch.ones(64,32,32).cuda(), #2 
            torch.ones(64,32,32).cuda(), #3
            torch.ones(64,32,32).cuda(), #4
            torch.ones(64,32,32).cuda(), #5
            torch.ones(64,32,32).cuda(), #6
            torch.ones(128,16,16).cuda(), #7
            torch.ones(128,16,16).cuda(), #8
            torch.ones(128,16,16).cuda(), #9
            torch.ones(128,16,16).cuda(), #10
            torch.ones(256,8,8).cuda(), #11
            torch.ones(256,8,8).cuda(), #12
            torch.ones(256,8,8).cuda(), #13
            torch.ones(256,8,8).cuda(), #14
            torch.ones(512,4,4).cuda(), #15
            torch.ones(512,4,4).cuda(), #16
            torch.ones(512,4,4).cuda(), #17
            torch.ones(512).cuda()
            ]
        for k in self.scale:
            k.requires_grad = True

    def forward(self, x):
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        
        out = F.relu(self.bn1(self.conv1(x * self.scale[0])))

        out_conv = F.relu(self.bn2(self.conv2(out * self.scale[1])))
        out_conv = self.bn3(self.conv3(out_conv * self.scale[2]))
        out_conv += self.shortcut_bn1(self.shortcut1(out  * self.scale[1]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn4(self.conv4(out * self.scale[3])))
        out_conv = self.bn5(self.conv5(out_conv * self.scale[4]))
        out_conv += self.shortcut_bn2(self.shortcut2(out  * self.scale[3]))
        out = F.relu(out_conv)

        # print (out.size(), self.scale[5].size())
        out_conv = F.relu(self.bn6(self.conv6(out * self.scale[5])))
        out_conv = self.bn7(self.conv7(out_conv * self.scale[6]))
        out_conv += self.shortcut_bn3(self.shortcut3(out  * self.scale[5]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn8(self.conv8(out * self.scale[7])))
        out_conv = self.bn9(self.conv9(out_conv * self.scale[8]))
        out_conv += self.shortcut_bn4(self.shortcut4(out  * self.scale[7]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn10(self.conv10(out * self.scale[9])))
        out_conv = self.bn11(self.conv11(out_conv * self.scale[10]))
        out_conv += self.shortcut_bn5(self.shortcut5(out  * self.scale[9]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn12(self.conv12(out * self.scale[11])))
        out_conv = self.bn13(self.conv13(out_conv * self.scale[12]))
        out_conv += self.shortcut_bn6(self.shortcut6(out  * self.scale[11]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn14(self.conv14(out * self.scale[13])))
        out_conv = self.bn15(self.conv15(out_conv * self.scale[14]))
        out_conv += self.shortcut_bn7(self.shortcut7(out  * self.scale[13]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn16(self.conv16(out * self.scale[15])))
        out_conv = self.bn17(self.conv17(out_conv * self.scale[16]))
        out_conv += self.shortcut_bn8(self.shortcut8(out  * self.scale[15]))
        out = F.relu(out_conv)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out * self.scale[17])
        return out


class ResNet34_inputmap(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34_inputmap, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.shortcut1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn1 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.shortcut2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn2 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.shortcut3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn3 = nn.BatchNorm2d(64)

        self.conv8 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(128)
        self.shortcut4 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        self.shortcut_bn4 = nn.BatchNorm2d(128)

        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(128)
        self.shortcut5 = nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn5 = nn.BatchNorm2d(128)

        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(128)
        self.conv13 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(128)
        self.shortcut6 = nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn6 = nn.BatchNorm2d(128)

        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(128)
        self.shortcut7 = nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn7 = nn.BatchNorm2d(128)

        self.conv16 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(256)
        self.conv17 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(256)
        self.shortcut8 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        self.shortcut_bn8 = nn.BatchNorm2d(256)

        self.conv18 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18 = nn.BatchNorm2d(256)
        self.conv19 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn19 = nn.BatchNorm2d(256)
        self.shortcut9 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn9 = nn.BatchNorm2d(256)

        self.conv20 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(256)
        self.conv21 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn21 = nn.BatchNorm2d(256)
        self.shortcut10 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn10 = nn.BatchNorm2d(256)

        self.conv22 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(256)
        self.conv23 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(256)
        self.shortcut11 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn11 = nn.BatchNorm2d(256)

        self.conv24 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn24 = nn.BatchNorm2d(256)
        self.conv25 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn25 = nn.BatchNorm2d(256)
        self.shortcut12 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn12 = nn.BatchNorm2d(256)

        self.conv26 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.BatchNorm2d(256)
        self.conv27 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn27 = nn.BatchNorm2d(256)
        self.shortcut13 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn13 = nn.BatchNorm2d(256)

        self.conv28 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn28 = nn.BatchNorm2d(512)
        self.conv29 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn29 = nn.BatchNorm2d(512)
        self.shortcut14 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.shortcut_bn14 = nn.BatchNorm2d(512)

        self.conv30 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn30 = nn.BatchNorm2d(512)
        self.conv31 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(512)
        self.shortcut15 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn15 = nn.BatchNorm2d(512)

        self.conv32 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(512)
        self.conv33 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn33 = nn.BatchNorm2d(512)
        self.shortcut16 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn16 = nn.BatchNorm2d(512)

        self.linear = nn.Linear(512, num_classes)

        self.scale = [
            torch.ones(3,32,32).cuda(),  #0
            torch.ones(64,32,32).cuda(), #1 
            torch.ones(64,32,32).cuda(), #2
            torch.ones(64,32,32).cuda(), #3
            torch.ones(64,32,32).cuda(), #4
            torch.ones(64,32,32).cuda(), #5
            torch.ones(64,32,32).cuda(), #6
            torch.ones(64,32,32).cuda(), #7
            torch.ones(128,16,16).cuda(), #8
            torch.ones(128,16,16).cuda(), #9
            torch.ones(128,16,16).cuda(), #10
            torch.ones(128,16,16).cuda(), #11
            torch.ones(128,16,16).cuda(), #12
            torch.ones(128,16,16).cuda(), #13
            torch.ones(128,16,16).cuda(), #14
            torch.ones(128,16,16).cuda(), #15
            torch.ones(256,8,8).cuda(), #16
            torch.ones(256,8,8).cuda(), #17
            torch.ones(256,8,8).cuda(), #18
            torch.ones(256,8,8).cuda(), #19
            torch.ones(256,8,8).cuda(), #20
            torch.ones(256,8,8).cuda(), #21
            torch.ones(256,8,8).cuda(), #22
            torch.ones(256,8,8).cuda(), #23
            torch.ones(256,8,8).cuda(), #24
            torch.ones(256,8,8).cuda(), #25
            torch.ones(256,8,8).cuda(), #26
            torch.ones(256,8,8).cuda(), #27
            torch.ones(512,4,4).cuda(), #28
            torch.ones(512,4,4).cuda(), #29
            torch.ones(512,4,4).cuda(), #30
            torch.ones(512,4,4).cuda(), #31
            torch.ones(512,4,4).cuda(), #32
            torch.ones(512).cuda()
            ]
        for k in self.scale:
            k.requires_grad = True

    def forward(self, x):
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        
        out = F.relu(self.bn1(self.conv1(x * self.scale[0])))

        # 64
        out_conv = F.relu(self.bn2(self.conv2(out * self.scale[1])))
        out_conv = self.bn3(self.conv3(out_conv * self.scale[2]))
        out_conv += self.shortcut_bn1(self.shortcut1(out  * self.scale[1]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn4(self.conv4(out * self.scale[3])))
        out_conv = self.bn5(self.conv5(out_conv * self.scale[4]))
        out_conv += self.shortcut_bn2(self.shortcut2(out  * self.scale[3]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn6(self.conv6(out * self.scale[5])))
        out_conv = self.bn7(self.conv7(out_conv * self.scale[6]))
        out_conv += self.shortcut_bn3(self.shortcut3(out  * self.scale[5]))
        out = F.relu(out_conv)

        # 128
        out_conv = F.relu(self.bn8(self.conv8(out * self.scale[7])))
        out_conv = self.bn9(self.conv9(out_conv * self.scale[8]))
        out_conv += self.shortcut_bn4(self.shortcut4(out  * self.scale[7]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn10(self.conv10(out * self.scale[9])))
        out_conv = self.bn11(self.conv11(out_conv * self.scale[10]))
        out_conv += self.shortcut_bn5(self.shortcut5(out  * self.scale[9]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn12(self.conv12(out * self.scale[11])))
        out_conv = self.bn13(self.conv13(out_conv * self.scale[12]))
        out_conv += self.shortcut_bn6(self.shortcut6(out  * self.scale[11]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn14(self.conv14(out * self.scale[13])))
        out_conv = self.bn15(self.conv15(out_conv * self.scale[14]))
        out_conv += self.shortcut_bn7(self.shortcut7(out  * self.scale[13]))
        out = F.relu(out_conv)

        # 256
        out_conv = F.relu(self.bn16(self.conv16(out * self.scale[15])))
        out_conv = self.bn17(self.conv17(out_conv * self.scale[16]))
        out_conv += self.shortcut_bn8(self.shortcut8(out  * self.scale[15]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn18(self.conv18(out * self.scale[17])))
        out_conv = self.bn19(self.conv19(out_conv * self.scale[18]))
        out_conv += self.shortcut_bn9(self.shortcut9(out  * self.scale[17]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn20(self.conv20(out * self.scale[19])))
        out_conv = self.bn21(self.conv21(out_conv * self.scale[20]))
        out_conv += self.shortcut_bn10(self.shortcut10(out  * self.scale[19]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn22(self.conv22(out * self.scale[21])))
        out_conv = self.bn23(self.conv23(out_conv * self.scale[22]))
        out_conv += self.shortcut_bn11(self.shortcut11(out  * self.scale[21]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn24(self.conv24(out * self.scale[23])))
        out_conv = self.bn25(self.conv25(out_conv * self.scale[24]))
        out_conv += self.shortcut_bn12(self.shortcut12(out  * self.scale[23]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn26(self.conv26(out * self.scale[25])))
        out_conv = self.bn27(self.conv27(out_conv * self.scale[26]))
        out_conv += self.shortcut_bn13(self.shortcut13(out  * self.scale[25]))
        out = F.relu(out_conv)

        #512
        out_conv = F.relu(self.bn28(self.conv28(out * self.scale[27])))
        out_conv = self.bn29(self.conv29(out_conv * self.scale[28]))
        out_conv += self.shortcut_bn14(self.shortcut14(out  * self.scale[27]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn30(self.conv30(out * self.scale[29])))
        out_conv = self.bn31(self.conv31(out_conv * self.scale[30]))
        out_conv += self.shortcut_bn15(self.shortcut15(out  * self.scale[29]))
        out = F.relu(out_conv)

        out_conv = F.relu(self.bn32(self.conv32(out * self.scale[31])))
        out_conv = self.bn33(self.conv33(out_conv * self.scale[32]))
        out_conv += self.shortcut_bn16(self.shortcut16(out  * self.scale[31]))
        out = F.relu(out_conv)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out * self.scale[33])
        return out