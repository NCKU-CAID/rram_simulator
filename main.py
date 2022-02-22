import os
import argparse

import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import models_vgg as vgg
from utils import *
from models_vgg import *
from models_resnet import *
from models_vgg_inputmap import *
from models_resnet_inputmap import *



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




model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))

parser = argparse.ArgumentParser(description='RRAM Conductance-Aware Training')

parser.add_argument('--dataset', '-d', default='cifar10', type=str,
                    help='choose dataset')
parser.add_argument('--resume', default='none', type=str,
                    help='choose checkpoint file to resume')
parser.add_argument('--arch', default='vgg11_bn', type=str,
                    help='Specify NN model')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    metavar='LRD', help='lr decay (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-e', '--evaluate', default=0, type=int,
                    help='evaluate model on validation set')
parser.add_argument('--finetune', default=0, type=int,
                    help='finetune resume model')
parser.add_argument('--gpu', default="0", type=str, 
                    help='cuda set_device (default: 0)')
parser.add_argument('--save-dir', default='saved_models', type=str,
                    help='direcotry to save best models')

parser.add_argument('-q', '--is-quantize', default=0, type=int,
                    help='quantize mode')
parser.add_argument('-m', '--quantize-mode', default='linear', type=str,
                    help='quantize mode')
parser.add_argument('-w', '--quantize-width', default=8, type=int,
                    help='quantize width') 
parser.add_argument('--base', default=2, type=float,
                    help='quantize base')                
parser.add_argument('--shifting', default=0, type=float,
                    help='resistance drifting percentage. example:1.2')
parser.add_argument('--non-uniform', default=0, type=int,
                    help='non-uniform drifting. (default: 0)')
parser.add_argument('--mode', default='w', type=str,
                    help='Restoring mode. (g: Gradients, w:weights, default: w)')
parser.add_argument('--rate', default=0.1, type=float,
                    help='Retoring rate. (default: 0.1)')
parser.add_argument('--var', default=0, type=float,
                    help='Variation. (default: 0, example: 5)')
parser.add_argument('--var-after', default=0, type=int,
                    help='Train scale after variation. (default: 0)')
parser.add_argument('--scale-resume', default='none', type=str,
                    help='choose scale file to resume.')
parser.add_argument('--prune', default=0, type=int,
                    help='prune the hot area  = 1, otherwise = 0.')
parser.add_argument('--thermal', default=0, type=int,
                    help='thermal = 1 and nq = 0 do the thermal effect analysis.')
parser.add_argument('-nq', '--nonuniform-quantization', default=0, type=int,
                    help='nq = 0 and thermal = 1 do thermal effect analysis, thermal =0 and nq = 1 do nonuniform quantization after prunning.')
parser.add_argument('-tq', '--train_with_quantization', default=0, type=int,
                    help='train_with_quantization in finetune mode,')
parser.add_argument('--experiment_group', default=1, type=int,
                    help='0 for comparision group(original distribution), 1 for experiment group(big sum on cold, small sum on hot area).')
parser.add_argument('--stop_gradient', default=0, type=int,
                    help='1 when we need to retrain after pruning')

parser.add_argument('--placement', default=0, type=int,
                    help='1 when we need to do placement.')

parser.add_argument('--prune_ratio', default='0', type=str,
                    help='write the prune_ratio for all layer.')
parser.add_argument('--split', default=0, type=int,
                    help='1 for split .')

parser.add_argument('--downgrade', default=0, type=int,
                    help='1 for downgrade .')
parser.add_argument('--tile_pairing', default=0, type=int,
                    help='1 for tile pairing .')

parser.add_argument('--experiment', default=0, type=int,
                    help='1 for 4,5,6bit experiment, it means do not seperate 8 bit to 4bit 4bit .')
parser.add_argument('--testbit', default=0, type=int,
                    help='4bit, 5bit, 6bit')
parser.add_argument('--weight_sense', default=0, type=int,
                    help = ' do weight sensitivity test.')
parser.add_argument('--affect_ratio', default=0, type=int,
                    help = ' thermal affected ratio of weight sensitivity.')
parser.add_argument('--direct', default=0, type=str,
                    help='choose place direct, br=bottom right, bl=bottom left, tr=top right, tl= top left')
parser.add_argument('--remapping', default=0, type=int,
                    help='1 for remapping')
parser.add_argument('--high_t_prune_remap', default=0, type=int,
                    help='1 for prune and remap')
parser.add_argument('--do_temp', default = 0 , type = int,
                    help= 'temperature simulation for all model')

parser.add_argument('--temperature', default=0, type=int,
                    help= ' enter the temperature.')

parser.add_argument('--ga', default=0, type=int,
                    help = 'do ga is 1')
parser.add_argument('--compensation', default=0, type=int,
                    help = 'compensate 1')
parser.add_argument('--compensation_rate', default=0, type=float,
                    help = 'rate')

parser.add_argument('--Rmax', default=300000, type=int,
                    help = 'max resistance')
parser.add_argument('--cell_resolution', default=2, type=int,
                    help = 'cell resolution')
parser.add_argument('--output_weight', default=0, type=int,
                    help = 'output weight')
parser.add_argument('--quantize', default=0, type=int,
                    help='quantize mode')

class quantize_opts:
    def __init__(self, is_quantize, _mode = 'linear', _width = 4, _base = 2):
        self.is_quantize = is_quantize
        self.mode = 'none'
        self.width = -1
        self.base = -1
        if is_quantize:
            self.mode = _mode
            self.width = _width
            self.base = _base


class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2), #padding=2保证输入输出尺寸相同
            nn.ReLU(),      #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def case(arch):
    return {
        'resnet18' : ResNet18(),
        'resnet34' : ResNet34(),
        'resnet50' : ResNet50(),
        'mobilenet': models.mobilenet_v2(),
        'lenet'    : LeNet(),     
        'inputmap_vgg11_bn': VGG11_bn(), 
        'inputmap_vgg16_bn': VGG16_bn(),
        'inputmap_resnet18': ResNet18_inputmap(),
        'inputmap_resnet34': ResNet34_inputmap(),
        'alexnet' : Alexnet(),
    }.get(arch, 'error')


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def main():
    global args
    args = parser.parse_args()
    # torch.cuda.set_device(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    # torch.cuda.set_device(6)
    if args.base == 1.414:
        args.base = 2**(1/2)
    print('base:', args.base)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.arch[0:3] == 'vgg':
        net = vgg.__dict__[args.arch]()
        net.features = torch.nn.DataParallel(net.features)
        net= torch.nn.DataParallel(net)
    else:
        print('hehe')
        net = case(args.arch)
        net = torch.nn.DataParallel(net)
        print('========================================')
    
    #net = VGG11_bias()
    net.cuda()#把網路架構丟到cuda
    print(net)

    if os.path.isfile(args.resume):#找weight丟到網路架構裡面
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        #for key in list(checkpoint.keys()):
        #    print('key = ', key)
        #    if "module." in key:
        #        #print('lala')
        #        checkpoint[key.replace('module.','')] = checkpoint[key]
        #        del checkpoint[key]
        net.load_state_dict(checkpoint)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    if args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        #讀training data
        trainset = torchvision.datasets.CIFAR10(root = '../datasets/cifar10data', train = True,
                                                download = True, transform = transforms.Compose([
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop(32, 4),
                                                transforms.ToTensor(),
                                                normalize]))
        #讀testing data
        testset = torchvision.datasets.CIFAR10(root = '../datasets/cifar10data', train = False,
                                            download = True, transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize]))

    elif args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST('../datasets/mnist', train=True, download = True, 
                                              transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                             ]))

        testset = torchvision.datasets.MNIST('../datasets/mnist', train=False, transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                             ]))
                                            
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size,
                                            shuffle = True, num_workers = 128)
                                        
    testloader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size,
                                            shuffle = False, num_workers = 128)     
    
    
    ####For Variation#####
    if args.var and args.is_quantize and not args.shifting:
        Variation(net, args.var)
        if args.quantize_mode == 'exp':
            _suffix = '_quantize_' + args.quantize_mode + '_' + str(args.quantize_width) + 'bit_base' + str(args.base)
        elif args.quantize_mode == 'power':
            _suffix = '_quantize_' + args.quantize_mode + '_' + str(args.quantize_width) + 'bit_base' + str(args.base) 
        elif args.quantize_mode == 'linear':
            _suffix = '_quantize_' + args.quantize_mode + '_' + str(args.quantize_width) + 'bit'
        else:
            print ('unexpected quantize mode!')
            exit()
        torch.save(net.state_dict(),'./' + args.save_dir +'/' + args.arch + _suffix + '.pt')
        exit()
    ######################

    if args.evaluate: ##### inference step

        if args.var_after:
            assert args.scale_resume != 'none'
            scale = torch.load(args.scale_resume) 
            net.module.scale = scale

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = args.momentum, weight_decay = args.weight_decay)
        if args.is_quantize:
            qtz_opts = quantize_opts(is_quantize = True, _mode = args.quantize_mode, _width = args.quantize_width, _base = args.base)
        else:
            qtz_opts = quantize_opts(is_quantize = False)
        print ('====================Start Evaluation==========================')
        assert args.resume != 'none'
        



        if args.ga==1:
            ga_algorithm(net, args, testloader, criterion, optimizer, qtz_opts, args.shifting, args.non_uniform, args.rate, args.mode, args.var)



        if args.prune == 1:
            prune_ratio_list = [0,0.4,0.2,0.4,0.3,0.3,0.9,0]
            prune_layer(net, args, prune_ratio_list)
            pred = Eval(testloader, net, criterion, optimizer, -1, qtz_opts, args.shifting, args.non_uniform, args.rate, args.mode, args.var)
            torch.save(net.state_dict(),'./' + args.save_dir +'/' + args.arch + '_prune' + str(prune_ratio_list) + '%' +  '_ideal.pt')
            

        if args.placement:
            
            sector, len_tmp_list, redundant_heatlist, idx = placement(net, args)
            if args.prune ==1 :
                prune_layer(net, args)
                pred = Eval(testloader, net, criterion, optimizer, -1, qtz_opts, args.shifting, args.non_uniform, args.rate, args.mode, args.var)
                #torch.save(net.state_dict(),'./' + args.save_dir +'/' + args.arch + '_prune' + args.prune_ratio + '%' + str(args.testbit) + 'bit' +'.pt')
                torch.save(net.state_dict(),'./' + args.save_dir +'/' + args.arch + '_prune' + args.prune_ratio + '%' + str(args.testbit) + '.pt')
            

            if args.thermal==1 and args.nonuniform_quantization ==0:
                thermal_after_placement(net, args, sector)


            if args.split == 1:
                sector, place_idx = split_after_prune(net, sector, args)
                thermal_after_split(net, sector, args, len_tmp_list, place_idx)
           
            if args.downgrade == 1:
                downgrading_8bit(net, args, sector, idx)

            #if args.tile_pairing ==1: ## 跑thermal全圖
            #    tile_pairing(net, sector, args, len_tmp_list, redundant_heatlist, idx)
            if args.remapping ==1:
                remapping_thermal(net, sector, args, len_tmp_list, redundant_heatlist, idx)
        
            if args.compensation == 1:
                compensation(net, sector, args, idx)
                #tile_pairing(net, sector, args, len_tmp_list, redundant_heatlist, idx)
        
        else:
            if args.weight_sense == 1:
                weight_sensitivity(net, args)
            if args.do_temp == 1:
                model_acc_under_thermal_impact(net, args)
            #pred = Eval(testloader, net, criterion, optimizer, -1, qtz_opts, args.shifting, args.non_uniform, args.rate, args.mode, args.var)

        if args.quantize:
            net_quantized = quantize_8bit(net, args.quantize_width)

        if args.output_weight:
            print_weight(net_quantized, args.quantize_width, args.cell_resolution, args.Rmax)
       


        print ('====================Start Evaluation==========================')
        pred = Eval(testloader, net, criterion, optimizer, -1, qtz_opts, args.shifting, args.non_uniform, args.rate, args.mode, args.var)
        #print_size_of_model(net)

        #if args.remapping ==0:
        #    #data = open('remap_split_data.txt','a')
        #    data = open('high_temp_remap_split_data.txt','a')
        #    #data = open('normal_temp_remap_split_data.txt','a')
        #    data.write('architechure = ' + args.arch + ', prune_ratio = ' + str(args.prune_ratio) + ', bit = ' + str(args.testbit) + ', direction = ' + args.direct + ', acc = ' + str(pred) + '\n')
        #if args.remapping ==1:
        #    #data = open('prune_remap_data.txt','a')
        #    if args.high_t_prune_remap == 1:
        #        data = open('normal_temp_prune_remap_data.txt','a')
        #    if args.high_t_prune_remap == 0:
        #        data = open('normal_temp_remap_data.txt','a')
        #    data.write('architechure = ' + args.arch + ', prune_ratio = ' + str(args.prune_ratio) + ', bit = ' + str(args.testbit) + ', direction = ' + args.direct + ', acc = ' + str(pred) + '\n')
    
    
    
    
    elif args.finetune: ###### finetune step
        assert args.resume != 'none'
        # print ('====================Original Accuracy=========================')
        # qtz_opts = quantize_opts(is_quantize = False)
        # pred = Eval(testloader, net, -1, qtz_opts)

        # print ('==============================================================')
        criterion = nn.CrossEntropyLoss().cuda()
        if args.var_after:
            optimizer = optim.SGD(net.module.scale, lr = 0.001, momentum = args.momentum, weight_decay = args.weight_decay)
        else:
            optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = args.momentum, weight_decay = args.weight_decay)
        best_pred = 0

        # ================================Quantize options============================
        if args.is_quantize:
            qtz_opts = quantize_opts(is_quantize = True, _mode = args.quantize_mode, _width = args.quantize_width, _base = args.base)
            if args.quantize_mode == 'exp':
                _suffix = '_quantize_' + args.quantize_mode + '_' + str(args.quantize_width) + 'bit_base' + str(args.base)
            elif args.quantize_mode == 'power':
                _suffix = '_quantize_' + args.quantize_mode + '_' + str(args.quantize_width) + 'bit_base' + str(args.base) 
            elif args.quantize_mode == 'linear':
                _suffix = '_quantize_' + args.quantize_mode + '_' + str(args.quantize_width) + 'bit'
            else:
                print ('unexpected quantize mode!')
                exit()
            print ('============ Quantize to ', args.quantize_width, 'bits under ' ,args.quantize_mode, ' mode ============')
        else:
            qtz_opts = quantize_opts(is_quantize = False)
            _suffix = '_original'
        
        # ================================Start training============================
        
        for epoch in range(int(args.epochs)):
            prune_ratio_list = [0,0.4,0.2,0.4,0.3,0.3,0.9,0.]
            Train(trainloader, net, criterion, optimizer, epoch, qtz_opts, args.shifting, args.non_uniform, args.var, args) ## original prune
            #Train(trainloader, net, criterion, optimizer, epoch, qtz_opts, args.shifting, args.non_uniform, args.var, args, prune_ratio_list)
            pred = Eval(testloader, net, criterion, optimizer, epoch, qtz_opts, args.shifting, args.non_uniform, args.rate, args.mode, args.var)
            
            
            if pred > best_pred:
                best_pred = pred
                #torch.save(net.state_dict(),'./' + args.save_dir + '/' + args.arch +  '_stop_gradient_' + args.prune_ratio + args.quantize_mode + str(args.quantize_width) + str(args.testbit) + '%.pt')
                torch.save(net.state_dict(),'./' + args.save_dir + '/' + args.arch +  '_retrain_' + str(prune_ratio_list)  + '%_ideal.pt')
           

            
            #if pred > best_pred:
            #    best_pred = pred
            #    torch.save(net.state_dict(),'./' + args.save_dir +'/' + args.arch + _suffix + '.pt')

            #    if args.var_after:
            #        torch.save(net.module.scale, './' + args.save_dir + '/scale_' + args.arch + _suffix + '.pt')
            ####### finetune is to sqweeze to be more high accrurate
            ####### So, we change the learning rate from 0.5 to 0.3.
            ####### It meansi that we update the gradient slowly
            if (epoch+1) % 20 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.3
                    
        print(best_pred)
    
    else:#### Training step
        criterion = nn.CrossEntropyLoss().cuda()
        if args.var_after:
            optimizer = optim.SGD(net.module.scale, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
        else:
            optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
        best_pred = 0

        # ================================Quantize options============================
        if args.is_quantize:
            qtz_opts = quantize_opts(is_quantize = True, _mode = args.quantize_mode, _width = args.quantize_width, _base = args.base)
            if args.quantize_mode == 'exp':
                _suffix = '_quantize_' + args.quantize_mode + '_' + str(args.quantize_width) + 'bit_base' + str(args.base) 
            elif args.quantize_mode == 'power':
                _suffix = '_quantize_' + args.quantize_mode + '_' + str(args.quantize_width) + 'bit_base' + str(args.base) 
            elif args.quantize_mode == 'linear':
                _suffix = '_quantize_' + args.quantize_mode + '_' + str(args.quantize_width) + 'bit'
            else:
                print ('unexpected quantize mode!')
                exit()
            print ('========= Quantize to ', args.quantize_width, 'bits under ' ,args.quantize_mode, ' mode =========')
        else:
            qtz_opts = quantize_opts(is_quantize = False)
            _suffix = '_original'
        
        # ================================Start training============================

        for epoch in range(args.epochs):
            Train(trainloader, net, criterion, optimizer, epoch, qtz_opts, args.shifting, args.non_uniform, args.var, args)
            pred = Eval(testloader, net, criterion, optimizer, epoch, qtz_opts, args.shifting, args.non_uniform, args.rate, args.mode, args.var)
            

            if pred > best_pred:
                best_pred = pred
                torch.save(net.state_dict(),'./' + args.save_dir +'/' + args.arch + _suffix +'.pt')

            if (epoch+1) % 30 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                    
        print(best_pred)
    
if __name__ == '__main__':
    main()
    # torch.save(net,'/home/zhu-z14/ljl/vgg/vgg_final.tar')
