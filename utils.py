import numpy as np
import math
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random
import copy
from random import choice

from fxpmath import Fxp

def quantize_8bit(model, width):
    max_weight = -9999.0
    min_weight = 9999.0
    model_backup = copy.deepcopy(model)
    for (ind, p) in enumerate(model_backup.parameters()):
    # for (ind, p) in enumerate(model.parameters()):
        if len(p.data.size()) == 4 or len(p.data.size()) == 2:
            w = p.data
            if(max_weight < w.max()):
                max_weight = w.max()
            if(min_weight > w.min()):
                min_weight = w.min()

    # print("max_weight = ", max_weight)
    # print("min_weight = ", min_weight)

    q_step_pos = (2**(width) - 1) / (max_weight)
    q_step_neg = (2**(width) - 1) / (min_weight) * -1

    # print("q_step_pos = ", q_step_pos)
    # print("q_step_neg = ", q_step_neg)
    # print("quantization_width = ", width)

    for (ind, p) in enumerate(model_backup.parameters()):
    # for (ind, p) in enumerate(model.parameters()):
        if len(p.data.size()) == 4 or len(p.data.size()) == 2:
            w = p.data
            w_pos = (w>0).float()*w
            w_neg = (w<0).float()*w*(-1)
            # print("************************************")
            # print("max_weight = ", w.max())
            # print("min_weight = ", w.min())

            w_pos = (w_pos * q_step_pos).round()
            w_neg = (w_neg * q_step_neg).round()

            w = (w_pos + (-1)*w_neg)
            # print("max_weight = ", w.max())
            # print("min_weight = ", w.min())

            p.data = w

    return model_backup

def print_weight(model, width, cell_resolution, Rmax):
    new_file = open("weight_file.txt", mode = "w")
    model_backup = copy.deepcopy(model)
    i = 1
    for (ind, p) in enumerate(model_backup.parameters()):
        if len(p.data.size()) == 4: #len==4 is mean convolution layer
            # new_file.write("convolution_layer_")
            # new_file.write("%d" % i)
            # new_file.write("_positive\n")
            matx = p.data.cpu().numpy()
            for (index1, value1) in enumerate (matx):
                for (index2, value2) in enumerate (value1):
                    for (index3, value3) in enumerate (value2):
                        for (index4, value4) in enumerate (value3):
                            # value4 = value4.astype(int)
                            value4 = value4.astype(float)
                            if(value4 > 0):
                                if((width / cell_resolution) == 1 or (width / cell_resolution) == 0):
                                    new_file.write("%f " % value4)
                                else:
                                    if(width > 3 * cell_resolution):
                                        MSB = (value4 % (2**(cell_resolution * 4))) / (2**(cell_resolution * 3))
                                        new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                                    if(width > 2 * cell_resolution):
                                        MSB = (value4 % (2**(cell_resolution * 3))) / (2**(cell_resolution * 2))
                                        new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                                    MSB = (value4 % (2**(cell_resolution * 2))) / (2**cell_resolution)
                                    new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                                    MSB = value4 % (2**cell_resolution)
                                    new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                            else:
                                if((width / cell_resolution) == 1 or (width / cell_resolution) == 0):
                                    new_file.write("0 ")
                                else:
                                    if(width > 3 * cell_resolution):
                                        new_file.write("0 ")
                                    if(width > 2 * cell_resolution):
                                        new_file.write("0 ")
                                    new_file.write("0 ")
                                    new_file.write("0 ")
            new_file.write("\n")
            # new_file.write("convolution_layer_")
            # new_file.write("%d" % i)
            # new_file.write("_negative\n")
            for (index1, value1) in enumerate (matx):
                for (index2, value2) in enumerate (value1):
                    for (index3, value3) in enumerate (value2):
                        for (index4, value4) in enumerate (value3):
                            value4 = value4 * (-1)
                            # value4 = value4.astype(int)
                            value4 = value4.astype(float)
                            if(value4 > 0):
                                if((width / cell_resolution) == 1 or (width / cell_resolution) == 0):
                                    new_file.write("%f " % value4)
                                else:
                                    if(width > 3 * cell_resolution):
                                        MSB = (value4 % (2**(cell_resolution * 4))) / (2**(cell_resolution * 3))
                                        new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                                    if(width > 2 * cell_resolution):
                                        MSB = (value4 % (2**(cell_resolution * 3))) / (2**(cell_resolution * 2))
                                        new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                                    MSB = (value4 % (2**(cell_resolution * 2))) / (2**cell_resolution)
                                    new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                                    MSB = value4 % (2**cell_resolution)
                                    new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                            else:
                                if((width / cell_resolution) == 1 or (width / cell_resolution) == 0):
                                    new_file.write("0 ")
                                else:
                                    if(width > 3 * cell_resolution):
                                        new_file.write("0 ")
                                    if(width > 2 * cell_resolution):
                                        new_file.write("0 ")
                                    new_file.write("0 ")
                                    new_file.write("0 ")

            new_file.write("\n")
            i = i + 1

        if len(p.data.size()) == 2: #len==2 is mean fc layer
            # new_file.write("fully_connected_layer_")
            # new_file.write("%d" % i)
            # new_file.write("_positive\n")
            matx = p.data.cpu().numpy()
            for (index1, value1) in enumerate (matx):
                for (index2, value2) in enumerate (value1):
                    # value2 = value2.astype(int)
                    value2 = value2.astype(float)
                    if(value2 > 0):
                        if((width / cell_resolution) == 1 or (width / cell_resolution) == 0):
                            new_file.write("%f " % value2)
                        else:
                            if(width > 3 * cell_resolution):
                                MSB = (value2 % (2**(cell_resolution * 4))) / (2**(cell_resolution * 3))
                                new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                            if(width > 2 * cell_resolution):
                                MSB = (value2 % (2**(cell_resolution * 3))) / (2**(cell_resolution * 2))
                                new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                            MSB = (value2 % (2**(cell_resolution * 2))) / (2**cell_resolution)
                            new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                            MSB = value2 % (2**cell_resolution)
                            new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                    else:
                        if((width / cell_resolution) == 1 or (width / cell_resolution) == 0):
                            new_file.write("0 ")
                        else:
                            if(width > 3 * cell_resolution):
                                new_file.write("0 ")
                            if(width > 2 * cell_resolution):
                                new_file.write("0 ")
                            new_file.write("0 ")
                            new_file.write("0 ")
            new_file.write("\n")
            # new_file.write("fully_connected_layer_")
            # new_file.write("%d" % i)
            # new_file.write("_negative\n")
            for (index1, value1) in enumerate (matx):
                for (index2, value2) in enumerate (value1):
                    value2 = value2 * (-1)
                    # value2 = value2.astype(int)
                    value2 = value2.astype(float)
                    if(value2 > 0):
                        if((width / cell_resolution) == 1 or (width / cell_resolution) == 0):
                            new_file.write("%f " % value2)
                        else:
                            if(width > 3 * cell_resolution):
                                MSB = (value2 % (2**(cell_resolution * 4))) / (2**(cell_resolution * 3))
                                new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                            if(width > 2 * cell_resolution):
                                MSB = (value2 % (2**(cell_resolution * 3))) / (2**(cell_resolution * 2))
                                new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                            MSB = (value2 % (2**(cell_resolution * 2))) / (2**cell_resolution)
                            new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                            MSB = value2 % (2**cell_resolution)
                            new_file.write("%d " % (MSB * Rmax / ((2**cell_resolution) - 1)))
                    else:
                        if((width / cell_resolution) == 1 or (width / cell_resolution) == 0):
                            new_file.write("0 ")
                        else:
                            if(width > 3 * cell_resolution):
                                new_file.write("0 ")
                            if(width > 2 * cell_resolution):
                                new_file.write("0 ")
                            new_file.write("0 ")
                            new_file.write("0 ")

            new_file.write("\n")
            i = i + 1
    new_file.close

def ga_algorithm(net, args, testloader, criterion, optimizer, qtz_opts, shifting, non_uniform, rate, mode, var):
    print('ok')
    size = 30  ## number of population
    max_gen = 100 ## number of maximum generation
    crossoverRate= 0.9
    mutationRate = 0.1
    popu_list = []
    fitness_list = []
    prune_list = []
    acc_list = []
    population, fitness_array = initial_population(size, net, args, testloader, criterion, optimizer, qtz_opts, shifting, non_uniform, rate, mode, var)
    for i in range(max_gen):
        population, fitness_array, prune_ratio, acc = reproduction(population, fitness_array, crossoverRate, mutationRate, size, net, args, testloader, criterion, optimizer, qtz_opts, shifting, non_uniform, rate, mode, var)
        popu_list.append(population)
        fitness_list.append(fitness_array)
        prune_list.append(prune_ratio)
        acc_list.append(acc)
    for i in range(max_gen):
        print('=========================generation {}==================================='.format(i))
        print('population list len = ', len(popu_list))
        print('population[{}]={}'.format(i,popu_list[i]))
        print('fitness[{}]={}'.format(i, fitness_list[i]))
        print('prunelist[{}]={}'.format(i, prune_list[i]))
        print('acc_list[{}]={}'.format(i, acc_list[i]))

def initial_population(size, net, args, testloader, criterion, optimizer, qtz_opts, shifting, non_uniform, rate, mode, var):

    population = []
    for i in range(size): ## create population
        each_chromosome = []
        for j in range(8):
            if j < 6 or i==7:
                a = random.randint(0,5)
            else:
                a = random.randint(0,8)
            #print('a=',a)
            each_chromosome.append((a*10))

        population.append(each_chromosome)

    #print('population = ', population)

    fitness_array=[]
    prune_ratio_array=[]
    acc_array = []
    for i in range(size):
        if os.path.isfile(args.resume):#???weight????????????????????????
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        fitness_score, t_prune_ratio, acc = calculate_fitness(population[i], size, net, args, testloader, criterion, optimizer, qtz_opts, shifting, non_uniform, rate, mode, var)

        fitness_array.append(fitness_score) ### dict structure = {which population, fitness_score of which population}
        prune_ratio_array.append(t_prune_ratio)
        acc_array.append(acc)

    print('chromosome prune ratio list = ', population)
    print('fitness_score array = ', fitness_array)
    print('prune_ratio_array = ', prune_ratio_array)
    print('acc_array = ', acc_array)
    return population, fitness_array



def calculate_fitness(chromosome, size, net, args, testloader, criterion, optimizer, qtz_opts, shifting, non_uniform, rate, mode, var): ## size means which chromosome in population

    total_prune_ratio = prune_layer(net, args, chromosome, size)
    acc = Eval(testloader, net, criterion, optimizer, -1, qtz_opts, shifting, non_uniform, rate, mode, var)

    #print('prune_ratio = ', total_prune_ratio)
    #print('acc = ', acc)
    fitness_score = (total_prune_ratio)*0.9 + (acc)*0.1
    print('fitness_score = ', fitness_score)

    return fitness_score, total_prune_ratio, acc




def reproduction(population, fitness_array, crossoverRate, mutationRate, size, net, args, testloader, criterion, optimizer, qtz_opts, shifting, non_uniform, rate, mode, var):
    new_population = []
    new_fitness_array = []
    new_t_prune_ratio = []
    new_acc = []
    for i in range (len(population)):
        parent1 = selection(population, fitness_array)
        parent2 = selection(population, fitness_array)

        #print('in reproduction population = ', population)
        #print('in reproduction fitness arary = ', fitness_array)
        #print('parent1 = ', parent1)
        #print('parent2 = ', parent2)

        c_prob = round(random.uniform(0,1),2)
        print('crossover prob = ', c_prob)
        if c_prob < crossoverRate:
            print('do crossover!!')
            chromosome = crossover(parent1, parent2)
        else:
            chromosome = parent1

        prob = round(random.uniform(0,1),2)
        print('muation prob = ', prob)
        if prob < mutationRate :
            print('do mutation!!!')
            chromosome = mutation(chromosome)


        if os.path.isfile(args.resume):#???weight????????????????????????
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


        fitness_score, t_prune_ratio, acc = calculate_fitness(chromosome, size, net, args, testloader, criterion, optimizer, qtz_opts, shifting, non_uniform, rate, mode, var)

        new_population.append(chromosome)
        new_fitness_array.append(fitness_score)
        new_t_prune_ratio.append(t_prune_ratio)
        new_acc.append(acc)

        print('chromosome prune ratio list = ', new_population)
        print('fitness_score array = ', new_fitness_array)
        print('prune_ratio_array = ', new_t_prune_ratio)
        print('acc_array = ', new_acc)



    return new_population, new_fitness_array, new_t_prune_ratio, new_acc





def selection(population, fitness_array):
    _sum=0
    #print('in selection population = ', population)
    #print('in selection fitness array = ', fitness_array)
    for i in range (len(fitness_array)):

        _sum = _sum + int(fitness_array[i]*100)
    #print('sum = ', _sum)
    choose_idx = random.randint(0,int(_sum))
    #print('choose_idx=', choose_idx)

    accum=0
    ret_chromosome= []
    for i in range(len(fitness_array)):
        #print('accum = ', accum)
        if choose_idx >= accum and choose_idx < (accum + (fitness_array[i]*100)) :
            ret_chromosome = population[i]
            #print('choose_idx', choose_idx)
            #print('accum = ', accum)
            #print('population{}={}'.format(i,population[i]))
            #print('ret chromosome = ', ret_chromosome)
            return ret_chromosome
        else:
            accum = accum + (fitness_array[i]*100)

    #return ret_chromosome


def crossover(parent1, parent2):

    #print('parent 1 = ', parent1 , '    parent2 = ', parent2)
    cut_point = random.randint(0,len(parent1))
    #print('cut point = ', cut_point)
    head = parent1[0:cut_point]
    tail = parent2[cut_point:len(parent2)]
    #print('head = ', head)
    #print('tail = ', tail)

    merge = head+tail
    #print('merge = ', merge)
    return merge



def mutation(chromosome):
    mutate_point = random.randint(0,len(chromosome)-1)
    rand_ratio = random.randint(0,5)*10
    chromosome[mutate_point] = rand_ratio
    return chromosome


def Variation(model, var = 5):
    print('Variation: {}'.format(var))
    for (ind, p) in enumerate(model.parameters()):
#        print('Variation parameter ind = {} and p = {}'.format(ind,p))
        if len(p.data.size()) == 4 or len(p.data.size()) == 2: #len==4 is mean convolution layer,len==2 is mean fc layer
            matr = p.data
            variation = np.random.normal(0, matr.abs().cpu().numpy()/var, matr.shape)
            matr += torch.FloatTensor(variation).cuda()
            p.data = matr


def DynamicVariation(model, var = 5):
    print('Dynamic Variation: {}%'.format(100*float(1/var)))
    for (ind, p) in enumerate(model.parameters()):
        if len(p.data.size()) == 4 or len(p.data.size()) == 2:
            matr = p.data
            #################
            inputs = p.data
            print('index = {}'.format(ind))
            print(inputs.data.shape)
            tmp = matr.abs().cpu().numpy();
            print('tmp ')
            print(tmp.shape)

            #################
            variation = np.random.normal(0, matr.abs().cpu().numpy()/var, matr.shape)

            matr -= torch.FloatTensor(variation).cuda().abs() #different with +-
#            matr *= var
            p.data = matr


def map_and_extend_all_layer(_model):

    SynapseBit = 8
    CellBit = 4
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = -1
    RealMax = 1
    SubarrayRow = 128
    SubarrayCol = 128
    all_sum_list = []
    rram_partition_conv={}
    rram_partition_fc={}
    ######### to sequence all layer's subarray
    for name, param in (_model.named_parameters()):
        if len(param.data.size()) ==4 :
            print(name)
            print(len(param.data.size()))
            row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition_total_sum = {}
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition_conv[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition_conv[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value,name]
                all_sum_list.append(tmp)

        if len(param.data.size()) ==2 :
            print(name)
            print(len(param.data.size()))
            row = param.data.shape[1]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition_total_sum = {}
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition_fc[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition_fc[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value, name]
                all_sum_list.append(tmp)


    #all_sum_list = sorted(all_sum_list, key = lambda s:s[1])
    all_sum_list = sorted(all_sum_list, key = lambda s:s[1], reverse = True)
    ratio = 0.3
    pruned_subarray = math.ceil(len(all_sum_list)*ratio)
    print('pruned subarray = ', pruned_subarray)
    for i in range(pruned_subarray):
        key, value, layer_name = all_sum_list[i]





        if layer_name =='features.module.0.weight' or layer_name =='features.module.2.weight' or layer_name =='features.module.5.weight' or layer_name == 'features.module.10.weight' or layer_name == 'features.module.12.weight':
            for name,param in (_model.named_parameters()):
                if name == layer_name:
                    row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
                    w = param.data.reshape(param.data.shape[0],row)
                    if w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow].shape == torch.zeros(128,128).shape:
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(128,128)
                        param.data = w.reshape(param.data.shape)




        if layer_name == 'classifier.0.weight' or layer_name == 'classifier.2.weight':
            for name,param in (_model.named_parameters()):
                if name == layer_name:
                    row = param.data.shape[1]
                    w = param.data.reshape(param.data.shape[0],row)
                    if w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow].shape == torch.zeros(128,128).shape:
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(128,128)
                        param.data = w.reshape(param.data.shape)





def compensation(_model, sector, args, idx ):
    realmin=1000
    realmax=-9999
    for name, param in (_model.named_parameters()):
        print(name)
        if len(param.data.size()) != 1:
            w = param.data
            if torch.min(param.data)<realmin:
                realmin = torch.min(param.data)
            if torch.max(param.data)>realmax:
                realmax = torch.max(param.data)

    SynapseBit = args.testbit
    CellBit = args.testbit
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = realmin
    RealMax = realmax
    SubarrayRow = 128
    SubarrayCol = 128
    cnt=1


    compensate_n_list=[]
    layer_len_list=[]

    for name, param in (_model.named_parameters()):
        if len(param.data.size()) == 4:
             row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
             w = param.data.reshape(param.data.shape[0],row)
             numRow = math.ceil(row/SubarrayRow)
             numCol = math.ceil(param.data.shape[0]/SubarrayCol)
             rram_partition = {}
             rram_partition_total_sum = {}
             rram_partition_total_sum_list = []
             for i in range(numCol):
                 for j in range(numRow):
                     rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                     rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
             for key, value in rram_partition_total_sum.items():
                 tmp = [key, value]
                 rram_partition_total_sum_list.append(tmp)

             rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])

             compensate_n = int(len(rram_partition_total_sum_list)*args.compensation_rate)
             compensate_n_list.append(compensate_n)
             layer_len_list.append(int(len(rram_partition_total_sum_list)))

        if len(param.data.size()) == 2:
             row = param.data.shape[1]
             w = param.data.reshape(param.data.shape[0],row)
             numRow = math.ceil(row/SubarrayRow)
             numCol = math.ceil(param.data.shape[0]/SubarrayCol)
             rram_partition = {}
             rram_partition_total_sum = {}
             rram_partition_total_sum_list = []
             for i in range(numCol):
                 for j in range(numRow):
                     rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                     rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
             for key, value in rram_partition_total_sum.items():
                 tmp = [key, value]
                 rram_partition_total_sum_list.append(tmp)

             rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])

             compensate_n = int(len(rram_partition_total_sum_list)*args.compensation_rate)
             compensate_n_list.append(compensate_n)
             layer_len_list.append(int(len(rram_partition_total_sum_list)))

    print('compensate_n_list = ', compensate_n_list)
    print('layer_len_list = ', layer_len_list)

    compensate_n_cnt = 0
    comp_layer=[]

    for k in range(len(layer_len_list)):
        if k == 0 :
            comp_layer.append(compensate_n_list[k])
        else :
            comp_layer.append(layer_len_list[k-1]- compensate_n_list[k-1] + compensate_n_list[k] + comp_layer[k-1])

    print('comp_layer=', comp_layer)


    #######################3
    #while cnt < idx:
    #    print('cnt=',cnt)
    #    print('idx=',idx)
    ########################


    for name, param in (_model.named_parameters()):
        if len(param.data.size()) == 4:

            row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {}
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key, value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])



            #if name == 'module.features.module.0.weight':
            #    compensate_n = comp_layer[0]
            #if name == 'module.features.module.2.weight':
            #    compensate_n = comp_layer[1]
            #if name == 'module.features.module.5.weight':
            #    compensate_n = comp_layer[2]
            #if name == 'module.features.module.7.weight':
            #    compensate_n = comp_layer[3]
            #if name == 'module.features.module.10.weight':
            #    compensate_n = comp_layer[4]
            #if name == 'module.features.module.12.weight':
            #    compensate_n = comp_layer[5]

            for cnt in range(idx):
                if sector[cnt][4]==name:
                    print(name)
                    #if compensate_n_cnt < compensate_n:
                    #    #print('asdasdasdasdasdasjdaskdjlaskdjlaksdj')
                    #    key = sector[i][2]
                    #    tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                    #    heat_value = int(sector[cnt][1])*(2**CellBit/(2**4))
                    #    print('original heat = ', int(sector[cnt][1]))
                    #    print('new heat = ', heat_value)
                    #    for j in range(tmp_w.shape[0]):
                    #        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                    #        data_with_thermal_impact = (newdata>(2**CellBit-heat_value-1)).float() * (2**CellBit-heat_value-1)
                    #        data_without_thermal_impact = (newdata<=(2**CellBit-heat_value-1)).float() * tmp_w[j]
                    #        data_with_compensate = (newdata>(2**CellBit-heat_value-1)).float() * tmp_w[j]
                    #        resume_data = data_with_compensate + data_without_thermal_impact
                    #        #tmp_w[j] = ((resume_data-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                    #        tmp_w[j] = resume_data
                    #    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                    #    param.data = w.reshape(param.data.shape)
                    #
                    #    compensate_n_cnt = compensate_n_cnt + 1
                    #    print('len of rram partition totalt sum lisst = ', int(len(rram_partition_total_sum_list)))
                    #    print('compensate_rate = ', args.compensation_rate)
                    #    print('compensate_n = ', compensate_n)
                    #    print('compensate_cnt = ', compensate_n_cnt)
                    #else :
                    #    print('------------------------------------------------------------asdasdasdasd')
                    #    key = sector[i][2]
                    #    tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                    #    heat_value = int(sector[cnt][1])*(2**CellBit/(2**4))
                    #    print('original heat = ', int(sector[cnt][1]))
                    #    print('new heat = ', heat_value)
                    #    for j in range(tmp_w.shape[0]):
                    #        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                    #        #print(newdata)
                    #        data_with_thermal_impact = (newdata>(2**CellBit-heat_value-1)).float() * (2**CellBit-heat_value-1)
                    #        data_without_thermal_impact = (newdata<=(2**CellBit-heat_value-1)).float() * newdata
                    #        resume_data = data_with_thermal_impact + data_without_thermal_impact
                    #        tmp_w[j] = ((resume_data-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                    #    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                    #    compensate_n_cnt = compensate_n_cnt + 1
                    #    param.data = w.reshape(param.data.shape)


                    key = sector[i][2]
                    tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                    heat_value = int(sector[cnt][1])/16 * (2**CellBit)
                    print('original heat = ', int(sector[cnt][1]))
                    print('new heat = ', heat_value)
                    for j in range(tmp_w.shape[0]):
                        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                        #newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                        #print(newdata)
                        data_with_thermal_impact = (newdata>((2**CellBit)-heat_value-1)).float() * ((2**CellBit)-heat_value-1)
                        data_without_thermal_impact = (newdata<=((2**CellBit)-heat_value-1)).float() * newdata
                        resume_data = data_with_thermal_impact + data_without_thermal_impact
                        #D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                        tmp_w[j] = ((resume_data-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                        #tmp_w[j]=resume_data
                    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                    #compensate_n_cnt = compensate_n_cnt + 1

                    #cnt=cnt+1
            param.data = w.reshape(param.data.shape)

        if len(param.data.size()) == 2:
            #print('name = ', name)
            row = param.data.shape[1]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {}
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key, value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])

            #if name == 'module.classifier.0.weight':
            #    compensate_n = comp_layer[6]
            #if name == 'module.classifier.2.weight':
            #    compesnate_n = comp_layer[7]

            for cnt in range(idx):

                if sector[cnt][4]==name:
                    #if compensate_n_cnt < compensate_n:
                    #    key = sector[i][2]
                    #    tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                    #    heat_value = int(sector[cnt][1])*((2**CellBit)/(2**4))
                    #    print('original heat = ', int(sector[cnt][1]))
                    #    print('new heat = ', heat_value)
                    #    for j in range(tmp_w.shape[0]):
                    #        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                    #        data_with_thermal_impact = (newdata>(2**CellBit-heat_value-1)).float() * (2**CellBit-heat_value-1)
                    #        data_without_thermal_impact = (newdata<=(2**CellBit-heat_value-1)).float() * tmp_w[j]
                    #        data_with_compensate = (newdata>(2**CellBit-heat_value-1)).float() * tmp_w[j]
                    #        resume_data = data_with_compensate + data_without_thermal_impact
                    #
                    #        #tmp_w[j] = ((resume_data-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                    #        tmp_w[j] = resume_data
                    #    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                    #    compensate_n_cnt = compensate_n_cnt + 1
                    #    param.data = w.reshape(param.data.shape)
                    #else :
                    #    print('===================================================asdasdasdasdasd')
                    #    key = sector[i][2]
                    #    tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                    #    heat_value = int(sector[cnt][1])*((2**CellBit)/(2**4))
                    #    print('original heat = ', int(sector[cnt][1]))
                    #    print('new heat = ', heat_value)
                    #    for j in range(tmp_w.shape[0]):
                    #        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                    #        data_with_thermal_impact = (newdata>(2**CellBit-heat_value-1)).float() * (2**CellBit-heat_value-1)
                    #        data_without_thermal_impact = (newdata<=(2**CellBit-heat_value-1)).float() * newdata
                    #        resume_data = data_with_thermal_impact + data_without_thermal_impact
                    #        tmp_w[j] = ((resume_data-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                    #    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                    #    compensate_n_cnt = compensate_n_cnt + 1
                    #    param.data = w.reshape(param.data.shape)

                    key = sector[i][2]
                    tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                    heat_value = int(sector[cnt][1])/16*(2**CellBit)
                    print('original heat = ', int(sector[cnt][1]))
                    print('new heat = ', heat_value)
                    for j in range(tmp_w.shape[0]):
                        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                        #print(newdata)
                        data_with_thermal_impact = (newdata>((2**CellBit)-heat_value-1)).float() * ((2**CellBit)-heat_value-1)
                        data_without_thermal_impact = (newdata<=((2**CellBit)-heat_value-1)).float() * newdata
                        #D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                        resume_data = data_with_thermal_impact + data_without_thermal_impact
                        tmp_w[j] = ((resume_data-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                        #tmp_w[j] = resume_data
                    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                    #compensate_n_cnt = compensate_n_cnt + 1


                    #cnt=cnt+1

            param.data = w.reshape(param.data.shape)



def thermal_after_placement(_model, args, sector):

    realmin=1000
    realmax=-9999
    for name, param in (_model.named_parameters()):
        print(name)
        if len(param.data.size()) != 1:
            w = param.data
            if torch.min(param.data)<realmin:
                realmin = torch.min(param.data)
            if torch.max(param.data)>realmax:
                realmax = torch.max(param.data)

    SynapseBit = 8
    CellBit = 4
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = realmin
    RealMax = realmax
    SubarrayRow = 128
    SubarrayCol = 128

    for name, param in (_model.named_parameters()):
        if len(param.data.size()) == 4:
            row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            col = param.data.shape[0]*numColPerSynapse
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)

            ### for alexnet
            if args.arch == 'alexnet':
                if name == 'module.features.0.weight' or name=='module.features.6.weight' or name=='module.features.8.weight' or name=='module.features.10.weight' or name=='module.classifier.0.weight' or name=='module.classifier.2.weight' or name=='module.classifier.4.weight':
                    for i in range(len(sector)):
                        if len(sector[i])== 7 and sector[i][4] == name :
                            key = sector[i][2]
                            first_or_second = sector[i][6]
                            tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                            first_sub = tmp_w[0:64, 0:128]
                            second_sub = tmp_w[64:128, 0:128]
                            first_sub_affect = torch.zeros(first_sub.shape)
                            second_sub_affect = torch.zeros(second_sub.shape)
                            if first_or_second == 1:
                                extend_sub = torch.zeros(int(first_sub.shape[0]*2), int(first_sub.shape[1]))
                                for j in range(first_sub.shape[0]):
                                    newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(first_sub[j]-RealMax) + NormalizedMax).round()
                                    value = newdata
                                    cellrange = 2**CellBit
                                    for k in range(numColPerSynapse):
                                        reminder = torch.ceil(value%cellrange)
                                        value = torch.floor(value/cellrange)
                                        extend_sub[(numColPerSynapse)*j+k] = reminder

                                heat_level= int(sector[i][1])

                                for j in range(0, extend_sub.shape[0], (numColPerSynapse)):
                                    if numColPerSynapse == 2:
                                        data_hot = (extend_sub[j]>(2**CellBit-heat_level-1)).float() * (2**CellBit-heat_level-1) + (extend_sub[j+1]>(2**CellBit-heat_level-1)).float() * (2**CellBit-heat_level-1) * (2**CellBit)
                                        data_cold = (extend_sub[j]<=(2**CellBit-heat_level-1)).float() * extend_sub[j] + (extend_sub[j+1]<=(2**CellBit-heat_level-1)).float() * extend_sub[j+1] * (2**CellBit)

                                    resumedata = data_hot + data_cold
                                    first_sub_affect[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                            else:
                                if second_sub.shape[0]!=0:
                                    extend_sub = torch.zeros(int(second_sub.shape[0]*2), int(second_sub.shape[1]))
                                    for j in range(second_sub.shape[0]):
                                        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(second_sub[j]-RealMax) + NormalizedMax).round()
                                        value = newdata
                                        cellrange = 2**CellBit
                                        for k in range(numColPerSynapse):
                                            reminder = torch.ceil(value%cellrange)
                                            value = torch.floor(value/cellrange)
                                            extend_sub[(numColPerSynapse)*j+k] = reminder
                                    heat_level = int(sector[i][1])
                                    for j in range(0, extend_sub.shape[0], (numColPerSynapse)):
                                        if numColPerSynapse == 2:
                                            data_hot = (extend_sub[j]>(2**CellBit-heat_level-1)).float() * (2**CellBit-heat_level-1) + (extend_sub[j+1]>(2**CellBit-heat_level-1)).float() * (2**CellBit-heat_level-1) * (2**CellBit)
                                            data_cold = (extend_sub[j]<=(2**CellBit-heat_level-1)).float() * extend_sub[j] + (extend_sub[j+1]<=(2**CellBit-heat_level-1)).float() * extend_sub[j+1] * (2**CellBit)

                                        resumedata = data_hot + data_cold
                                        second_sub_affect[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+RealMax


                            tmp_w[0:64, 0:128] = first_sub_affect
                            tmp_w[64:128, 0:128] = second_sub_affect
                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w


                            param.data = w.reshape(param.data.shape)


            #### for VGG11
            if args.arch == 'vgg11':
                if name == 'module.features.module.0.weight' or name == 'module.features.module.3.weight'or name=='module.features.module.6.weight' or name=='module.features.module.8.weight' or name=='module.features.module.11.weight' or name=='module.features.module.13.weight' or name=='module.features.module.16.weight' or name == 'module.features.module.18.weight':
                    for i in range(len(sector)):
                        if len(sector[i])== 7 and sector[i][4] == name :
                            key = sector[i][2]
                            first_or_second = sector[i][6]
                            tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                            first_sub = tmp_w[0:64, 0:128]
                            second_sub = tmp_w[64:128, 0:128]
                            first_sub_affect = torch.zeros(first_sub.shape)
                            second_sub_affect = torch.zeros(second_sub.shape)
                            if first_or_second == 1:
                                extend_sub = torch.zeros(int(first_sub.shape[0]*2), int(first_sub.shape[1]))
                                for j in range(first_sub.shape[0]):
                                    newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(first_sub[j]-RealMax) + NormalizedMax).round()
                                    value = newdata
                                    cellrange = 2**CellBit
                                    for k in range(numColPerSynapse):
                                        reminder = torch.ceil(value%cellrange)
                                        value = torch.floor(value/cellrange)
                                        extend_sub[(numColPerSynapse)*j+k] = reminder

                                heat_level= int(sector[i][1])

                                for j in range(0, extend_sub.shape[0], (numColPerSynapse)):
                                    if numColPerSynapse == 2:
                                        data_hot = (extend_sub[j]>(2**CellBit-heat_level-1)).float() * (2**CellBit-heat_level-1) + (extend_sub[j+1]>(2**CellBit-heat_level-1)).float() * (2**CellBit-heat_level-1) * (2**CellBit)
                                        data_cold = (extend_sub[j]<=(2**CellBit-heat_level-1)).float() * extend_sub[j] + (extend_sub[j+1]<=(2**CellBit-heat_level-1)).float() * extend_sub[j+1] * (2**CellBit)

                                    resumedata = data_hot + data_cold
                                    first_sub_affect[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                            else:
                                if second_sub.shape[0]!=0:
                                    extend_sub = torch.zeros(int(second_sub.shape[0]*2), int(second_sub.shape[1]))
                                    for j in range(second_sub.shape[0]):
                                        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(second_sub[j]-RealMax) + NormalizedMax).round()
                                        value = newdata
                                        cellrange = 2**CellBit
                                        for k in range(numColPerSynapse):
                                            reminder = torch.ceil(value%cellrange)
                                            value = torch.floor(value/cellrange)
                                            extend_sub[(numColPerSynapse)*j+k] = reminder
                                    heat_level = int(sector[i][1])
                                    for j in range(0, extend_sub.shape[0], (numColPerSynapse)):
                                        if numColPerSynapse == 2:
                                            data_hot = (extend_sub[j]>(2**CellBit-heat_level-1)).float() * (2**CellBit-heat_level-1) + (extend_sub[j+1]>(2**CellBit-heat_level-1)).float() * (2**CellBit-heat_level-1) * (2**CellBit)
                                            data_cold = (extend_sub[j]<=(2**CellBit-heat_level-1)).float() * extend_sub[j] + (extend_sub[j+1]<=(2**CellBit-heat_level-1)).float() * extend_sub[j+1] * (2**CellBit)

                                        resumedata = data_hot + data_cold
                                        second_sub_affect[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+RealMax


                            tmp_w[0:64, 0:128] = first_sub_affect
                            tmp_w[64:128, 0:128] = second_sub_affect
                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w


                            param.data = w.reshape(param.data.shape)

        if len(param.data.size()) ==2:
            row = param.data.shape[1]
            col = param.data.shape[0]*numColPerSynapse
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)


            ###### for vgg11 and alexnet###############
            if args.arch == 'alexnet' or args.arch == 'vgg11':
                if name=='module.classifier.0.weight' or name=='module.classifier.2.weight' or name=='module.classifier.4.weight':
                    for i in range(len(sector)):
                        if len(sector[i])== 7 and sector[i][4] == name :
                            key = sector[i][2]
                            first_or_second = sector[i][6]
                            tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                            first_sub = tmp_w[0:64, 0:128]
                            second_sub = tmp_w[64:128, 0:128]
                            first_sub_affect = torch.zeros(first_sub.shape)
                            second_sub_affect = torch.zeros(second_sub.shape)
                            if first_or_second == 1:
                                extend_sub = torch.zeros(int(first_sub.shape[0]*2), int(first_sub.shape[1]))
                                for j in range(first_sub.shape[0]):
                                    newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(first_sub[j]-RealMax) + NormalizedMax).round()
                                    value = newdata
                                    cellrange = 2**CellBit
                                    for k in range(numColPerSynapse):
                                        reminder = torch.ceil(value%cellrange)
                                        value = torch.floor(value/cellrange)
                                        extend_sub[(numColPerSynapse)*j+k] = reminder

                                heat_level= int(sector[i][1])

                                for j in range(0, extend_sub.shape[0], (numColPerSynapse)):
                                    if numColPerSynapse == 2:
                                        data_hot = (extend_sub[j]>(2**CellBit-heat_level-1)).float() * (2**CellBit-heat_level-1) + (extend_sub[j+1]>(2**CellBit-heat_level-1)).float() * (2**CellBit-heat_level-1) * (2**CellBit)
                                        data_cold = (extend_sub[j]<=(2**CellBit-heat_level-1)).float() * extend_sub[j] + (extend_sub[j+1]<=(2**CellBit-heat_level-1)).float() * extend_sub[j+1] * (2**CellBit)

                                    resumedata = data_hot + data_cold
                                    first_sub_affect[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                            else:
                                if second_sub.shape[0]!=0:
                                    extend_sub = torch.zeros(int(second_sub.shape[0]*2), int(second_sub.shape[1]))
                                    for j in range(second_sub.shape[0]):
                                        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(second_sub[j]-RealMax) + NormalizedMax).round()
                                        value = newdata
                                        cellrange = 2**CellBit
                                        for k in range(numColPerSynapse):
                                            reminder = torch.ceil(value%cellrange)
                                            value = torch.floor(value/cellrange)
                                            extend_sub[(numColPerSynapse)*j+k] = reminder
                                    heat_level = int(sector[i][1])
                                    for j in range(0, extend_sub.shape[0], (numColPerSynapse)):
                                        if numColPerSynapse == 2:
                                            data_hot = (extend_sub[j]>(2**CellBit-heat_level-1)).float() * (2**CellBit-heat_level-1) + (extend_sub[j+1]>(2**CellBit-heat_level-1)).float() * (2**CellBit-heat_level-1) * (2**CellBit)
                                            data_cold = (extend_sub[j]<=(2**CellBit-heat_level-1)).float() * extend_sub[j] + (extend_sub[j+1]<=(2**CellBit-heat_level-1)).float() * extend_sub[j+1] * (2**CellBit)

                                        resumedata = data_hot + data_cold
                                        second_sub_affect[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+RealMax


                            tmp_w[0:64, 0:128] = first_sub_affect
                            tmp_w[64:128, 0:128] = second_sub_affect
                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w


                            param.data = w.reshape(param.data.shape)




def thermal_effect(_model, args, sector):



    realmin=1000
    realmax=-9999
    for name, param in (_model.named_parameters()):
        if len(param.data.size()) != 1:
            w = param.data

            if torch.min(param.data)<realmin:
                realmin = torch.min(param.data)
            if torch.max(param.data)>realmax:
                realmax = torch.max(param.data)

    print('realmin = ', realmin)
    print('realmax = ', realmax)



    SynapseBit = 8
    CellBit = 4
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = realmin
    RealMax = realmax
    SubarrayRow = 128
    SubarrayCol = 128
    print('test')

    #level = 3

    for name, param in (_model.named_parameters()):
        if len(param.data.size()) == 4 :
            row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            col = param.data.shape[0]*numColPerSynapse
            w = param.data.reshape(param.data.shape[0],row)

            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_row_sum = {} ## a row sum in subarray
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)


            if args.experiment_group:
                rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])
                #rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1], reverse = True)
            else:
                print('tt')
                random.shuffle(rram_partition_total_sum_list)


            print('name = ', name)
            print('rram_partition total sum list = ', rram_partition_total_sum_list)



            ############# distribution 1 is a better distribution, distribution 2 is a worse distribution

            if name == 'features.module.2.weight':

                ####### subarray heat distribution1 #####
                #different_level = 2
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 6
                #        level = 2
                #    if indx == 1:
                #        front = 6
                #        end = 9
                #        level = 1
                ####################################


                ##### subarray heat distribution1 prune and retrain  #####
                different_level = 3
                front=0
                end=0
                for indx in range(different_level):
                    if indx == 0:
                        front = 0
                        end = 4
                        level = 0
                    if indx == 1:
                        front = 4
                        end = 6
                        level = 6
                    if indx == 2:
                        front = 6
                        end = 9
                        level = 5
                ##################################



                ####### subarray heat distribution2 ####
                #different_level = 1
                #front = 0
                #end =0
                #for indx in range(different_level):
                #    if indx==0:
                #        front =0
                #        end =9
                #        level=5
                #########################################
                ####### subarray heat distribution2 after prunning and retrain####
                #different_level = 2
                #front = 0
                #end = 0
                #for indx in range(different_level):
                #    if indx==0:
                #        front = 0
                #        end = 4
                #        level = 0
                #    if indx==1:
                #        front = 4
                #        end = 9
                #        level = 5
                #########################################






                    for i in range(front,end):
                        key,value = rram_partition_total_sum_list[i]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]


                        rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            ## normalized data
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            for k in range(numColPerSynapse):
                                reminder = torch.ceil(value%cellrange)
                                value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                                rram_map[(numColPerSynapse)*j+k] = reminder


                        w_prune = torch.zeros(tmp_w.shape)
                        for j in range(0,rram_map.shape[0],(numColPerSynapse)):


                            ### this is for weightbit = 8 cellbit =4
                            if numColPerSynapse == 2:
                                resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) * (2**CellBit)
                                resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-level-1)).float() * rram_map[j+1] * (2**CellBit)

                            ### this is for weightbit =8 cellbit = 2
                            if numColPerSynapse == 4:
                                resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float()*(2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit) + (rram_map[j+2]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)*(2**CellBit)
                                resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float()*(rram_map[j]) + (rram_map[j+1]<=(2**CellBit-level-1)).float()*(rram_map[j+1])*(2**CellBit) + (rram_map[j+2]<=(2**CellBit-level-1)).float()*(rram_map[j+2])*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]<=(2**CellBit-level-1)).float()*(rram_map[j+3])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                            if numColPerSynapse == 8:
                                resumedata_prune = 0
                                resumedata_noprune = rram_map[j] + rram_map[j+1]*(2**CellBit) + rram_map[j+2]*(2**CellBit)*(2**CellBit) + rram_map[j+3]*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+4]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+5]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+6]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+7]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)




                            resumedata = resumedata_prune + resumedata_noprune
                            w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune


                        #print('after = prune number {} = {}'.format(i,w_prune))
                param.data = w.reshape(param.data.shape)


            if name == 'features.module.5.weight':
                ratio = 1
                level = 1
                pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)



                ####### subarray heat distribution1 #####
                #different_level = 2
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 15
                #        level = 2
                #    if indx == 1:
                #        front = 15
                #        end = 18
                #        level = 1
                ##########################################
                ###### subarray heat distribution1 prune and retrain #####
                different_level = 3
                front=0
                end=0
                for indx in range(different_level):
                    if indx == 0:
                        front = 0
                        end = 9
                        level = 0
                    if indx == 1:
                        front = 9
                        end = 15
                        level = 6
                    if indx == 2:
                        front = 15
                        end = 18
                        level = 5
                #########################################


                ###### subarray heat distribution2 #####
                #different_level = 1
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx==0:
                #        front=0
                #        end=18
                #        level=5
                ##########################################
                ###### subarray heat distribution2 after pruning and retrain
                #different_level = 2
                #front =0
                #end = 0
                #for indx in range(different_level):
                #    if indx==0:
                #        front= 0
                #        end= 9
                #        level= 0
                #    if indx==0:
                #        front= 9
                #        end=18
                #        level= 5
                #################################################################



                    for i in range(front,end):
                        key,value = rram_partition_total_sum_list[i]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]


                        rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            ## normalized data
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            for k in range(numColPerSynapse):
                                reminder = torch.ceil(value%cellrange)
                                value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                                rram_map[(numColPerSynapse)*j+k] = reminder



                        w_prune = torch.zeros(tmp_w.shape)
                        for j in range(0,rram_map.shape[0],(numColPerSynapse)):

                            ### this is for weightbit = 8 cellbit =4
                            if numColPerSynapse == 2:
                                resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) * (2**CellBit)
                                resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-level-1)).float() * rram_map[j+1] * (2**CellBit)

                            ### this is for weightbit =8 cellbit = 2
                            if numColPerSynapse == 4:
                                resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float()*(2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit) + (rram_map[j+2]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)*(2**CellBit)
                                resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float()*(rram_map[j]) + (rram_map[j+1]<=(2**CellBit-level-1)).float()*(rram_map[j+1])*(2**CellBit) + (rram_map[j+2]<=(2**CellBit-level-1)).float()*(rram_map[j+2])*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]<=(2**CellBit-level-1)).float()*(rram_map[j+3])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                            if numColPerSynapse == 8:
                                resumedata_prune = 0
                                resumedata_noprune = rram_map[j] + rram_map[j+1]*(2**CellBit) + rram_map[j+2]*(2**CellBit)*(2**CellBit) + rram_map[j+3]*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+4]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+5]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+6]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+7]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)



                            resumedata = resumedata_prune + resumedata_noprune
                            w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune


                        #print('after = prune number {} = {}'.format(i,w_prune))
                param.data = w.reshape(param.data.shape)



            if name == 'features.module.7.weight':


                ####### subarray heat distribution1 #####
                #different_level = 2
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 24
                #        level = 2
                #    if indx == 1:
                #        front = 24
                #        end = 36
                #        level = 1
                ###########################################
                 ###### subarray heat distribution1 #####
                different_level = 3
                front=0
                end=0
                for indx in range(different_level):
                    if indx == 0:
                        front = 0
                        end = 18
                        level = 0
                    if indx == 1:
                        front = 18
                        end = 28
                        level = 6
                    if indx == 2:
                        front = 28
                        end = 36
                        level = 5
                ##########################################


                ###### subarray heat distribution2 #####
                #different_level = 4
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx ==0:
                #        front=0
                #        end=15
                #        level=5
                #    if indx ==1:
                #        front=15
                #        end=22
                #        level=4
                #    if indx ==2:
                #        front=22
                #        end=27
                #        level=3
                #    if indx==3:
                #        front=27
                #        end=36
                #        level=2
                ################################################

                ######## subarray heat distribution2 after pruning and retrain####
                #different_level = 4
                #front = 0
                #end =0
                #for indx in range(different_level):
                #    if indx==0:
                #        front= 0
                #        end = 18
                #        level= 0
                #    if indx==1:
                #        front= 18
                #        end= 21
                #        level= 4
                #    if indx==2:
                #        front= 21
                #        end= 26
                #        level= 3
                #    if indx== 3:
                #        front= 26
                #        end= 36
                #        level= 2
                ##################################################################




                    for i in range(front,end):
                        #print('i=', i)
                        key,value = rram_partition_total_sum_list[i]
                        #print('key = ', key)
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]


                        rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            ## normalized data
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            for k in range(numColPerSynapse):
                                reminder = torch.ceil(value%cellrange)
                                value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                                rram_map[(numColPerSynapse)*j+k] = reminder



                        w_prune = torch.zeros(tmp_w.shape)
                        for j in range(0,rram_map.shape[0],(numColPerSynapse)):

                            ### this is for weightbit = 8 cellbit =4
                            if numColPerSynapse == 2:
                                resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) * (2**CellBit)
                                resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-level-1)).float() * rram_map[j+1] * (2**CellBit)

                            ### this is for weightbit =8 cellbit = 2
                            if numColPerSynapse == 4:
                                resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float()*(2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit) + (rram_map[j+2]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)*(2**CellBit)
                                resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float()*(rram_map[j]) + (rram_map[j+1]<=(2**CellBit-level-1)).float()*(rram_map[j+1])*(2**CellBit) + (rram_map[j+2]<=(2**CellBit-level-1)).float()*(rram_map[j+2])*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]<=(2**CellBit-level-1)).float()*(rram_map[j+3])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                            if numColPerSynapse == 8:
                                resumedata_prune = 0
                                resumedata_noprune = rram_map[j] + rram_map[j+1]*(2**CellBit) + rram_map[j+2]*(2**CellBit)*(2**CellBit) + rram_map[j+3]*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+4]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+5]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+6]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+7]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)



                            resumedata = resumedata_prune + resumedata_noprune
                            w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune


                        #print('after = prune number {} = {}'.format(i,w_prune))
                param.data = w.reshape(param.data.shape)

            if name == 'features.module.10.weight':



                ####### subarray heat distribution1 #####
                #different_level = 3
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 38
                #        level = 2
                #    if indx == 1:
                #        front = 38
                #        end = 50
                #        level = 1
                #    if indx == 2:
                #        front = 50
                #        end = 72
                #        level = 0
                ##########################################

                ###### subarray heat distribution1 prune and retrain #####
                different_level = 3
                front=0
                end=0
                for indx in range(different_level):
                    if indx == 0:
                        front = 0
                        end = 36
                        level = 0
                    if indx == 1:
                        front = 36
                        end = 54
                        level = 5
                    if indx == 2:
                        front = 54
                        end = 72
                        level = 4
                #########################################

                ###### subarray heat distribution2 ######
                #different_level = 5
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx ==0:
                #        front=0
                #        end=13
                #        level=5
                #    if indx ==1:
                #        front=13
                #        end=15
                #        level=4
                #    if indx ==2:
                #        front=15
                #        end=17
                #        level=3
                #    if indx==3:
                #        front=17
                #        end=23
                #        level=2
                #    if indx==4:
                #        front=23
                #        end=72
                #        level=1
                ##############################################

                ###### subarray heat distribution2 after prune and retrain ######
                #different_level=2
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx==0:
                #        front= 0
                #        end= 36
                #        level= 0
                #    if indx==1:
                #        front= 36
                #        end= 72
                #        level= 1
                ##################################################################


                    for i in range(front,end):
                        key,value = rram_partition_total_sum_list[i]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]


                        rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            ## normalized data
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            for k in range(numColPerSynapse):
                                reminder = torch.ceil(value%cellrange)
                                value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                                rram_map[(numColPerSynapse)*j+k] = reminder

                        w_prune = torch.zeros(tmp_w.shape)
                        for j in range(0,rram_map.shape[0],(numColPerSynapse)):
                            #print('rram_map = ', rram_map)

                           ### this is for weightbit = 8 cellbit =4
                            if numColPerSynapse == 2:
                                resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) * (2**CellBit)
                                resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-level-1)).float() * rram_map[j+1] * (2**CellBit)

                            ### this is for weightbit =8 cellbit = 2
                            if numColPerSynapse == 4:
                                resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float()*(2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit) + (rram_map[j+2]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)*(2**CellBit)
                                resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float()*(rram_map[j]) + (rram_map[j+1]<=(2**CellBit-level-1)).float()*(rram_map[j+1])*(2**CellBit) + (rram_map[j+2]<=(2**CellBit-level-1)).float()*(rram_map[j+2])*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]<=(2**CellBit-level-1)).float()*(rram_map[j+3])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                            if numColPerSynapse == 8:
                                resumedata_prune = 0
                                resumedata_noprune = rram_map[j] + rram_map[j+1]*(2**CellBit) + rram_map[j+2]*(2**CellBit)*(2**CellBit) + rram_map[j+3]*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+4]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+5]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+6]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+7]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)



                            resumedata = resumedata_prune + resumedata_noprune
                            #print('resumedata = ', resumedata)
                            w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune


                        #print('after = prune number {} = {}'.format(i,w_prune))
                param.data = w.reshape(param.data.shape)

            if name == 'features.module.12.weight':

                ####### subarray heat distribution1 #####
                #different_level = 4
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 9
                #        level = 4
                #    if indx == 1:
                #        front = 9
                #        end = 17
                #        level = 3
                #    if indx == 2:
                #        front = 17
                #        end = 45
                #        level = 2
                #    if indx == 3:
                #        front = 45
                #        end = 61
                #        level = 1
                #    if indx == 4:
                #        front = 61
                #        end = 144
                #        level = 0
                #############################################

                ###### subarray heat distribution1 prune and retrain #####
                different_level = 2
                front=0
                end=0
                for indx in range(different_level):
                    if indx == 0:
                        front = 0
                        end = 72
                        level = 0
                    if indx == 1:
                        front = 72
                        end = 144
                        level = 4
                ############################################


                ###### subarray heat distribution2 #####
                #different_level = 4
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 20
                #        level = 5
                #    if indx == 1:
                #        front = 20
                #        end = 24
                #        level = 4
                #    if indx == 2:
                #        front = 24
                #        end = 37
                #        level = 3
                #    if indx == 3:
                #        front = 37
                #        end = 54
                #        level = 2
                #    if indx == 4:
                #        front = 54
                #        end = 144
                #        level = 1
                #########################################


                ####### subarray heat distribution2 after pruning and retrain#######
                #different_level = 2
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx==0:
                #        front= 0
                #        end= 72
                #        level= 0
                #    if indx==1:
                #        front= 72
                #        end= 144
                #        level= 1
                ###################################################################




                    for i in range(front,end):
                        key,value = rram_partition_total_sum_list[i]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]


                        rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            ## normalized data
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            for k in range(numColPerSynapse):
                                reminder = torch.ceil(value%cellrange)
                                value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                                rram_map[(numColPerSynapse)*j+k] = reminder


                        w_prune = torch.zeros(tmp_w.shape)
                        for j in range(0,rram_map.shape[0],(numColPerSynapse)):

                            ### this is for weightbit = 8 cellbit =4
                            if numColPerSynapse == 2:
                                resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) * (2**CellBit)
                                resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-level-1)).float() * rram_map[j+1] * (2**CellBit)

                            ### this is for weightbit =8 cellbit = 2
                            if numColPerSynapse == 4:
                                resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float()*(2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit) + (rram_map[j+2]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)*(2**CellBit)
                                resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float()*(rram_map[j]) + (rram_map[j+1]<=(2**CellBit-level-1)).float()*(rram_map[j+1])*(2**CellBit) + (rram_map[j+2]<=(2**CellBit-level-1)).float()*(rram_map[j+2])*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]<=(2**CellBit-level-1)).float()*(rram_map[j+3])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                            if numColPerSynapse == 8:
                                resumedata_prune = 0
                                resumedata_noprune = rram_map[j] + rram_map[j+1]*(2**CellBit) + rram_map[j+2]*(2**CellBit)*(2**CellBit) + rram_map[j+3]*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+4]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+5]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+6]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+7]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)



                            resumedata = resumedata_prune + resumedata_noprune
                            w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune


                        #print('after = prune number {} = {}'.format(i,w_prune))
                param.data = w.reshape(param.data.shape)






        if len(param.data.size()) == 2 :
            row = param.data.shape[1]
            col = param.data.shape[0]*numColPerSynapse
            w = param.data.reshape(param.data.shape[0],row)

            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_row_sum = {} ## a row sum in subarray
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)

            if args.experiment_group:
                rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])
                #rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1], reverse = True)
            else:
                random.shuffle(rram_partition_total_sum_list)

            if name == 'classifier.0.weight':



                ######### subarray heat distribution 1 #######
                #different_level = 5
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 195
                #        level = 4
                #    if indx == 1:
                #        front = 195
                #        end = 273
                #        level = 3
                #    if indx == 2:
                #        front = 273
                #        end = 329
                #        level = 2
                #    if indx == 3:
                #        front = 329
                #        end = 390
                #        level = 1
                #    if indx == 4:
                #        front = 390
                #        end = 512
                #        level = 0
                # ############################################

                ######## subarray heat distribution 1 prune and retrain #######
                different_level = 5
                front=0
                end=0
                for indx in range(different_level):
                    if indx == 0:
                        front = 0
                        end = 256
                        level = 0
                    if indx == 1:
                        front = 256
                        end = 271
                        level = 7
                    if indx == 2:
                        front = 271
                        end = 334
                        level = 6
                    if indx == 3:
                        front = 334
                        end = 396
                        level = 5
                    if indx == 4:
                        front = 396
                        end = 512
                        level = 4
                 ############################################


                ####### subarray heat distribution 2 ########
                #different_level = 5
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 122
                #        level = 5
                #    if indx == 1:
                #        front = 122
                #        end = 192
                #        level = 4
                #    if indx == 2:
                #        front = 192
                #        end = 332
                #        level = 3
                #    if indx == 3:
                #        front = 332
                #        end = 419
                #        level = 2
                #    if indx == 4:
                #        front = 419
                #        end = 512
                #        level = 1
                ################################################


                ####### subarray heat distribution2 after prune and retrain #####
                #different_level =4
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx==0:
                #        front= 0
                #        end= 256
                #        level= 0
                #    if indx==1:
                #        front= 256
                #        end= 332
                #        level= 3
                #    if indx==2:
                #        front= 332
                #        end= 419
                #        level= 2
                #    if indx==3:
                #        front= 419
                #        end= 512
                #        level= 1
                #################################################################


                    for i in range(front,end):
                        key,value = rram_partition_total_sum_list[i]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]


                        rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            ## normalized data
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            for k in range(numColPerSynapse):
                                reminder = torch.ceil(value%cellrange)
                                value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                                rram_map[(numColPerSynapse)*j+k] = reminder


                        w_prune = torch.zeros(tmp_w.shape)
                        for j in range(0,rram_map.shape[0],(numColPerSynapse)):

                             ### this is for weightbit = 8 cellbit =4
                            if numColPerSynapse == 2:
                                resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) * (2**CellBit)
                                resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-level-1)).float() * rram_map[j+1] * (2**CellBit)

                            ### this is for weightbit =8 cellbit = 2
                            if numColPerSynapse == 4:
                                resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float()*(2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit) + (rram_map[j+2]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)*(2**CellBit)
                                resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float()*(rram_map[j]) + (rram_map[j+1]<=(2**CellBit-level-1)).float()*(rram_map[j+1])*(2**CellBit) + (rram_map[j+2]<=(2**CellBit-level-1)).float()*(rram_map[j+2])*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]<=(2**CellBit-level-1)).float()*(rram_map[j+3])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                            if numColPerSynapse == 8:
                                resumedata_prune = 0
                                resumedata_noprune = rram_map[j] + rram_map[j+1]*(2**CellBit) + rram_map[j+2]*(2**CellBit)*(2**CellBit) + rram_map[j+3]*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+4]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+5]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+6]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+7]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)


                            resumedata = resumedata_prune + resumedata_noprune
                            w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune


                param.data = w.reshape(param.data.shape)


            if name == 'classifier.2.weight':

                ######### subarray heat distribution 1 ####################
                #different_level = 2
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 3
                #        level = 2
                #    if indx == 1:
                #        front = 3
                #        end = 8
                #        level = 1
                ###########################################################

                ######## subarray heat distribution 1 prune and retrain####################
                different_level = 2
                front=0
                end=0
                for indx in range(different_level):
                    if indx == 0:
                        front = 0
                        end = 4
                        level = 0
                    if indx == 1:
                        front = 4
                        end = 8
                        level = 5
                ##########################################################


                ######## subarray heat distribution 2 ###################
                #different_level = 2
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 5
                #        level = 5
                #    if indx == 1:
                #        front = 5
                #        end = 8
                #        level = 4
                ########################################################


                ######## subarray heat distribution 2 after prune and retrain##???
                #different_level = 3
                #front= 0
                #end=0
                #for indx in range(different_level):
                #    if indx==0:
                #        front= 0
                #        end= 4
                #        level= 0
                #    if indx==1:
                #        front= 4
                #        end= 5
                #        level= 5
                #    if indx==2:
                #        front= 5
                #        end= 8
                #        level= 4

                ##############################################################

                    for i in range(front,end):
                        key,value = rram_partition_total_sum_list[i]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]


                        rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            ## normalized data
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            for k in range(numColPerSynapse):
                                reminder = torch.ceil(value%cellrange)
                                value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                                rram_map[(numColPerSynapse)*j+k] = reminder


                        w_prune = torch.zeros(tmp_w.shape)
                        for j in range(0,rram_map.shape[0],(numColPerSynapse)):

                            ### this is for weightbit = 8 cellbit =4
                            if numColPerSynapse == 2:
                                resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) * (2**CellBit)
                                resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-level-1)).float() * rram_map[j+1] * (2**CellBit)

                            ### this is for weightbit =8 cellbit = 2
                            if numColPerSynapse == 4:
                                resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float()*(2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit) + (rram_map[j+2]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)*(2**CellBit)
                                resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float()*(rram_map[j]) + (rram_map[j+1]<=(2**CellBit-level-1)).float()*(rram_map[j+1])*(2**CellBit) + (rram_map[j+2]<=(2**CellBit-level-1)).float()*(rram_map[j+2])*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]<=(2**CellBit-level-1)).float()*(rram_map[j+3])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                            if numColPerSynapse == 8:
                                resumedata_prune = 0
                                resumedata_noprune = rram_map[j] + rram_map[j+1]*(2**CellBit) + rram_map[j+2]*(2**CellBit)*(2**CellBit) + rram_map[j+3]*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+4]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+5]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+6]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+7]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)



                            resumedata = resumedata_prune + resumedata_noprune
                            w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune


                param.data = w.reshape(param.data.shape)







def nonuniform_quantization(_model):
    print('===================================== start nonuniform quantization ===============================================')


    realmin=1000
    realmax=-9999
    for name, param in (_model.named_parameters()):
        if len(param.data.size()) != 1:
            w = param.data

            if torch.min(param.data)<realmin:
                realmin = torch.min(param.data)
            if torch.max(param.data)>realmax:
                realmax = torch.max(param.data)

    print('realmin = ', realmin)
    print('realmax = ', realmax)



    SynapseBit = 8
    CellBit = 4
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = realmin
    RealMax = realmax
    SubarrayRow = 128
    SubarrayCol = 128
    print('test')

    #level = 3

    for name, param in (_model.named_parameters()):

        if len(param.data.size()) == 4 :
            row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            col = param.data.shape[0]*numColPerSynapse
            w = param.data.reshape(param.data.shape[0],row)

            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_row_sum = {} ## a row sum in subarray
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])
            #rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1], reverse = True)

            if name == 'features.module.2.weight':


                ####### subarray heat distribution1 #####
                #different_level = 2
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 6
                #        level = 2
                #    if indx == 1:
                #        front = 6
                #        end = 9
                #        level = 1
                ####################################


                #### subarray heat distribution1 prune and retrain  #####
                #different_level = 3
                #front=0
                #end=0
                #n_q=0
                #cell_limit = 7
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 4
                #        level = 0
                #        n_q=0
                #    if indx == 1:
                #        front = 4
                #        end = 6
                #        level = 6
                #        n_q= 1
                #    if indx == 2:
                #        front = 6
                #        end = 9
                #        level = 5
                #        n_q = 1
                #################################

                ##### subarray heat distribution1 prune and retrain 4x4 sliding window quantization #####
                different_level = 3
                front=0
                end=0
                n_q=0
                cell_limit = 7
                for indx in range(different_level):
                    if indx == 0:
                        front = 0
                        end = 4
                        level = 0
                        n_q=0
                    if indx == 1:
                        front = 4
                        end = 6
                        level = 2
                        n_q= 1
                    if indx == 2:
                        front = 6
                        end = 9
                        level = 1
                        n_q = 1
                ##################################




                ######3 subarray heat distribution 2 ###############
                #different_level = 1
                #front=0
                #end=0
                #n_q = 0
                #cell_limit= 7
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 9
                #        level = 4
                #        n_q = 0
                ########################################################

                ####### subarray heat distribution2 after prunning and retrain####
                #different_level = 2
                #front = 0
                #end = 0
                #n_q = 0
                #cell_limit= 7
                #for indx in range(different_level):
                #    if indx==0:
                #        front = 0
                #        end = 4
                #        level = 0
                #        n_q = 0
                #    if indx==1:
                #        front = 4
                #        end = 9
                #        level = 4
                #        n_q = 0
                #########################################






                    for i in range(front,end):
                        key,value = rram_partition_total_sum_list[i]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        #print('key = {} , value = {}'.format(key, value))

                        if n_q == 0:
                            rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse), int(tmp_w.shape[1]))
                        if n_q == 1:
                            rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse)*2, int(tmp_w.shape[1]))
                        #rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            ## normalized data
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            if n_q == 0 :
                                for k in range(numColPerSynapse):
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                                    rram_map[(numColPerSynapse)*j+k] = reminder
                            if n_q == 1 :
                                for k in range(0,2*numColPerSynapse,2):
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange)
                                    rram_map1_big = (reminder > cell_limit).float() * cell_limit
                                    rram_map1_sml = (reminder <= cell_limit).float() * reminder
                                    rram_map2_big = (reminder > cell_limit).float() * (reminder-cell_limit)
                                    rram_map2_sml = (reminder <= cell_limit).float() * 0
                                    rram_map[(numColPerSynapse)*(j*2) + k] = rram_map1_big + rram_map1_sml
                                    rram_map[(numColPerSynapse)*(j*2) + (k+1)] = rram_map2_big + rram_map2_sml
                        w_prune = torch.zeros(tmp_w.shape)
                        if n_q == 0:
                            for j in range(0,rram_map.shape[0],(numColPerSynapse)):
                                ### this is for weightbit = 8 cellbit =4
                                if numColPerSynapse == 2:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) * (2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-level-1)).float() * rram_map[j+1] * (2**CellBit)

                                ### this is for weightbit =8 cellbit = 2
                                if numColPerSynapse == 4:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float()*(2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit) + (rram_map[j+2]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)*(2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float()*(rram_map[j]) + (rram_map[j+1]<=(2**CellBit-level-1)).float()*(rram_map[j+1])*(2**CellBit) + (rram_map[j+2]<=(2**CellBit-level-1)).float()*(rram_map[j+2])*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]<=(2**CellBit-level-1)).float()*(rram_map[j+3])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                                if numColPerSynapse == 8:
                                    resumedata_prune = 0
                                    resumedata_noprune = rram_map[j] + rram_map[j+1]*(2**CellBit) + rram_map[j+2]*(2**CellBit)*(2**CellBit) + rram_map[j+3]*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+4]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+5]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+6]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+7]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)



                                resumedata = resumedata_prune + resumedata_noprune
                                w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune

                        if n_q == 1:
                            for j in range(0,rram_map.shape[0], numColPerSynapse*2):
                                if numColPerSynapse == 2:
                                    #print('j=',j)
                                    #print('rram_map size = ', rram_map.shape)
                                    resumedata = (rram_map[j]+rram_map[j+1]) + (rram_map[j+2]+rram_map[j+3])*(2**CellBit)

                                if numColPerSynapse == 4:
                                    resumedata = (rram_map[j]+rram_map[j+1]) + (rram_map[j+2]+rram_map[j+3])*(2**CellBit) + (rram_map[j+3]+rram_map[j+4])*(2**CellBit)*(2**CellBit) + (rram_map[j+5]+rram_map[j+6])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                                if numColPerSynapse == 8:
                                    print('8 bit weight with 1 bit cell resolution can not use nonuniform quantization !!')



                                w_prune[int(j/(numColPerSynapse*2))] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax
                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune



                param.data = w.reshape(param.data.shape)




            if name == 'features.module.5.weight':


                ####### subarray heat distribution1 #####
                #different_level = 2
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 15
                #        level = 2
                #    if indx == 1:
                #        front = 15
                #        end = 18
                #        level = 1
                ##########################################


                ###### subarray heat distribution1 prune and retrain #####
                #different_level = 3
                #front=0
                #end=0
                #n_q = 0
                #cell_limit=7
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 9
                #        level = 0
                #        n_q = 0
                #    if indx == 1:
                #        front = 9
                #        end = 15
                #        level = 6
                #        n_q = 1
                #    if indx == 2:
                #        front = 15
                #        end = 18
                #        level = 5
                #        n_q = 1
                #########################################

                ###### subarray heat distribution1 prune and retrain 4x4 sliding window quantization #####
                different_level = 4
                front=0
                end=0
                n_q = 0
                cell_limit=7
                for indx in range(different_level):
                    if indx == 0:
                        front = 0
                        end = 9
                        level = 0
                        n_q = 0
                    if indx == 1:
                        front = 9
                        end = 12
                        level = 2
                        n_q = 1
                    if indx == 2:
                        front = 12
                        end =15
                        level = 2
                        n_q = 1
                    if indx == 3:
                        front = 15
                        end = 18
                        level = 1
                        n_q = 1
                #########################################







                ########## subarray heat distribution 2 ###############
                #different_level = 1
                #front=0
                #end=0
                #n_q = 0
                #cell_limit= 7
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 18
                #        level = 4
                #        n_q = 0
                ###############################################

                ###### subarray heat distribution2 after pruning and retrain
                #different_level = 2
                #front =0
                #end = 0
                #n_q=0
                #cell_limit= 7
                #for indx in range(different_level):
                #    if indx==0:
                #        front= 0
                #        end= 9
                #        level= 0
                #        n_q= 0
                #    if indx==0:
                #        front= 9
                #        end= 18
                #        level= 4
                #        n_q= 1
                #################################################################




                    for i in range(front,end):
                        key,value = rram_partition_total_sum_list[i]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        #print('key = {} , value = {}'.format(key, value))

                        if n_q == 0:
                            rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse), int(tmp_w.shape[1]))
                        if n_q == 1:
                            rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse)*2, int(tmp_w.shape[1]))
                        #rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            ## normalized data
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            if n_q == 0 :
                                for k in range(numColPerSynapse):
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                                    rram_map[(numColPerSynapse)*j+k] = reminder
                            if n_q == 1 :
                                for k in range(0,2*numColPerSynapse,2):
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange)
                                    rram_map1_big = (reminder > cell_limit).float() * cell_limit
                                    rram_map1_sml = (reminder <= cell_limit).float() * reminder
                                    rram_map2_big = (reminder > cell_limit).float() * (reminder-cell_limit)
                                    rram_map2_sml = (reminder <= cell_limit).float() * 0
                                    rram_map[(numColPerSynapse)*(j*2) + k] = rram_map1_big + rram_map1_sml
                                    rram_map[(numColPerSynapse)*(j*2) + (k+1)] = rram_map2_big + rram_map2_sml
                        w_prune = torch.zeros(tmp_w.shape)
                        if n_q == 0:
                            for j in range(0,rram_map.shape[0],(numColPerSynapse)):
                                ### this is for weightbit = 8 cellbit =4
                                if numColPerSynapse == 2:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) * (2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-level-1)).float() * rram_map[j+1] * (2**CellBit)

                                ### this is for weightbit =8 cellbit = 2
                                if numColPerSynapse == 4:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float()*(2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit) + (rram_map[j+2]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)*(2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float()*(rram_map[j]) + (rram_map[j+1]<=(2**CellBit-level-1)).float()*(rram_map[j+1])*(2**CellBit) + (rram_map[j+2]<=(2**CellBit-level-1)).float()*(rram_map[j+2])*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]<=(2**CellBit-level-1)).float()*(rram_map[j+3])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                                if numColPerSynapse == 8:
                                    resumedata_prune = 0
                                    resumedata_noprune = rram_map[j] + rram_map[j+1]*(2**CellBit) + rram_map[j+2]*(2**CellBit)*(2**CellBit) + rram_map[j+3]*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+4]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+5]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+6]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+7]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)



                                resumedata = resumedata_prune + resumedata_noprune
                                w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune

                        if n_q == 1:
                            for j in range(0,rram_map.shape[0], numColPerSynapse*2):
                                if numColPerSynapse == 2:
                                    #print('j=',j)
                                    #print('rram_map size = ', rram_map.shape)
                                    resumedata = (rram_map[j]+rram_map[j+1]) + (rram_map[j+2]+rram_map[j+3])*(2**CellBit)

                                if numColPerSynapse == 4:
                                    resumedata = (rram_map[j]+rram_map[j+1]) + (rram_map[j+2]+rram_map[j+3])*(2**CellBit) + (rram_map[j+3]+rram_map[j+4])*(2**CellBit)*(2**CellBit) + (rram_map[j+5]+rram_map[j+6])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                                if numColPerSynapse == 8:
                                    print('8 bit weight with 1 bit cell resolution can not use nonuniform quantization !!')



                                w_prune[int(j/(numColPerSynapse*2))] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax
                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune



                param.data = w.reshape(param.data.shape)





            if name == 'features.module.7.weight':


                ####### subarray heat distribution1 #####
                #different_level = 2
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 24
                #        level = 2
                #    if indx == 1:
                #        front = 24
                #        end = 36
                #        level = 1
                ###########################################


                ###### subarray heat distribution1 prune and retrain#####
                #different_level = 3
                #front=0
                #end=0
                #n_q = 0
                #cell_limit = 7
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 18
                #        level = 0
                #        n_q = 0
                #    if indx == 1:
                #        front = 18
                #        end = 28
                #        level = 6
                #        n_q = 1
                #    if indx == 2:
                #        front = 28
                #        end = 36
                #        level = 5
                #        n_q = 1
                ##########################################

                ###### subarray heat distribution1 prune and retrain  4x4 sliding window quantization#####
                different_level = 4
                front=0
                end=0
                n_q = 0
                cell_limit = 7
                for indx in range(different_level):
                    if indx == 0:
                        front = 0
                        end = 18
                        level = 0
                        n_q = 0
                    if indx == 1:
                        front = 18
                        end = 22
                        level = 2
                        n_q = 1
                    if indx == 2:
                        front = 22
                        end = 28
                        level = 2
                        n_q = 1
                    if indx == 3:
                        front = 28
                        end = 36
                        level = 1
                        n_q = 1
                ##########################################


                ###### subarray heat distribution2  ########################
                #different_level = 4
                #front=0
                #end=0
                #n_q = 0
                #cell_limit= 7
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 15
                #        level = 4
                #        n_q = 0
                #    if indx == 1:
                #        front = 15
                #        end = 22
                #        level = 3
                #        n_q = 0
                #    if indx == 2:
                #        front = 22
                #        end = 28
                #        level = 2
                #        n_q = 0
                #    if indx ==3:
                #        front = 28
                #        end = 36
                #        level = 1
                #        n_q = 0
                #
                ##################################


                ######## subarray heat distribution2 after pruning and retrain####
                #different_level =4
                #front = 0
                #end =0
                #n_q=0
                #cell_limit= 7
                #for indx in range(different_level):
                #    if indx==0:
                #        front=0
                #        end = 18
                #        level= 0
                #        n_q= 0
                #    if indx==1:
                #        front= 18
                #        end= 21
                #        level= 3
                #        n_q= 0
                #    if indx==2:
                #        front= 21
                #        end= 26
                #        level= 2
                #        n_q= 0
                #    if indx==3:
                #        front= 26
                #        end= 36
                #        level= 1
                #        n_q= 0
                ##################################################################





                    for i in range(front,end):
                        key,value = rram_partition_total_sum_list[i]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        #print('key = {} , value = {}'.format(key, value))

                        if n_q == 0:
                            rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse), int(tmp_w.shape[1]))
                        if n_q == 1:
                            rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse)*2, int(tmp_w.shape[1]))
                        #rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            ## normalized data
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            if n_q == 0 :
                                for k in range(numColPerSynapse):
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                                    rram_map[(numColPerSynapse)*j+k] = reminder
                            if n_q == 1 :
                                for k in range(0,2*numColPerSynapse,2):
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange)
                                    rram_map1_big = (reminder > cell_limit).float() * cell_limit
                                    rram_map1_sml = (reminder <= cell_limit).float() * reminder
                                    rram_map2_big = (reminder > cell_limit).float() * (reminder-cell_limit)
                                    rram_map2_sml = (reminder <= cell_limit).float() * 0
                                    rram_map[(numColPerSynapse)*(j*2) + k] = rram_map1_big + rram_map1_sml
                                    rram_map[(numColPerSynapse)*(j*2) + (k+1)] = rram_map2_big + rram_map2_sml
                        w_prune = torch.zeros(tmp_w.shape)
                        if n_q == 0:
                            for j in range(0,rram_map.shape[0],(numColPerSynapse)):
                                ### this is for weightbit = 8 cellbit =4
                                if numColPerSynapse == 2:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) * (2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-level-1)).float() * rram_map[j+1] * (2**CellBit)

                                ### this is for weightbit =8 cellbit = 2
                                if numColPerSynapse == 4:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float()*(2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit) + (rram_map[j+2]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)*(2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float()*(rram_map[j]) + (rram_map[j+1]<=(2**CellBit-level-1)).float()*(rram_map[j+1])*(2**CellBit) + (rram_map[j+2]<=(2**CellBit-level-1)).float()*(rram_map[j+2])*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]<=(2**CellBit-level-1)).float()*(rram_map[j+3])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                                if numColPerSynapse == 8:
                                    resumedata_prune = 0
                                    resumedata_noprune = rram_map[j] + rram_map[j+1]*(2**CellBit) + rram_map[j+2]*(2**CellBit)*(2**CellBit) + rram_map[j+3]*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+4]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+5]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+6]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+7]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)



                                resumedata = resumedata_prune + resumedata_noprune
                                w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune

                        if n_q == 1:
                            for j in range(0,rram_map.shape[0], numColPerSynapse*2):
                                if numColPerSynapse == 2:
                                    #print('j=',j)
                                    #print('rram_map size = ', rram_map.shape)
                                    resumedata = (rram_map[j]+rram_map[j+1]) + (rram_map[j+2]+rram_map[j+3])*(2**CellBit)

                                if numColPerSynapse == 4:
                                    resumedata = (rram_map[j]+rram_map[j+1]) + (rram_map[j+2]+rram_map[j+3])*(2**CellBit) + (rram_map[j+3]+rram_map[j+4])*(2**CellBit)*(2**CellBit) + (rram_map[j+5]+rram_map[j+6])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                                if numColPerSynapse == 8:
                                    print('8 bit weight with 1 bit cell resolution can not use nonuniform quantization !!')



                                w_prune[int(j/(numColPerSynapse*2))] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax
                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune



                param.data = w.reshape(param.data.shape)


            if name == 'features.module.10.weight':

                ####### subarray heat distribution1 #####
                #different_level = 3
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 38
                #        level = 2
                #    if indx == 1:
                #        front = 38
                #        end = 50
                #        level = 1
                #    if indx == 2:
                #        front = 50
                #        end = 72
                #        level = 0
                ##########################################

                ###### subarray heat distribution1 prune and retrain #####
                #different_level = 3
                #front=0
                #end=0
                #n_q = 0
                #cell_limit = 7
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 36
                #        level = 0
                #        n_q = 0
                #    if indx == 1:
                #        front = 36
                #        end = 54
                #        level = 5
                #        n_q = 1
                #    if indx == 2:
                #        front = 54
                #        end = 72
                #        level = 4
                #        n_q = 0
                #########################################

                ###### subarray heat distribution1 prune and retrain 4x4 sliding window quantization#####
                different_level = 5
                front=0
                end=0
                n_q = 0
                cell_limit = 7
                for indx in range(different_level):
                    if indx == 0:
                        front = 0
                        end = 36
                        level = 0
                        n_q = 0
                    if indx == 1:
                        front = 36
                        end = 40
                        level = 1
                        n_q = 1
                    if indx == 2:
                        front = 40
                        end = 54
                        level = 1
                        n_q = 1
                    if indx == 3:
                        front = 54
                        end = 58
                        level = 0
                        n_q = 1
                    if indx == 4:
                        front = 58
                        end = 72
                        level = 0
                        n_q = 0
                #########################################




                ######## subarray heat distribution2 #####################
                #different_level = 4
                #front=0
                #end=0
                #n_q = 0
                #cell_limit= 7
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 2
                #        level = 3
                #        n_q = 0
                #    if indx == 1:
                #        front = 2
                #        end = 4
                #        level = 2
                #        n_q = 0
                #    if indx == 2:
                #        front = 4
                #        end = 10
                #        level = 1
                #        n_q = 0
                #    if indx == 3:
                #        front = 10
                #        end = 72
                #        level = 0
                #        n_q = 0
                ###################################################################


                ###### subarray heat distribution2 after prune and retrain ######
                #different_level=2
                #front=0
                #end=0
                #n_q=0
                #cell_limit=7
                #for indx in range(different_level):
                #    if indx==0:
                #        front= 0
                #        end= 36
                #        level= 0
                #        n_q= 0
                #    if indx==1:
                #        front= 36
                #        end= 72
                #        level= 0
                #        n_q= 0
                ##################################################################





                    for i in range(front,end):
                        key,value = rram_partition_total_sum_list[i]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        #print('key = {} , value = {}'.format(key, value))

                        if n_q == 0:
                            rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse), int(tmp_w.shape[1]))
                        if n_q == 1:
                            rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse)*2, int(tmp_w.shape[1]))
                        #rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            ## normalized data
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            if n_q == 0 :
                                for k in range(numColPerSynapse):
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                                    rram_map[(numColPerSynapse)*j+k] = reminder
                            if n_q == 1 :
                                for k in range(0,2*numColPerSynapse,2):
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange)
                                    rram_map1_big = (reminder > cell_limit).float() * cell_limit
                                    rram_map1_sml = (reminder <= cell_limit).float() * reminder
                                    rram_map2_big = (reminder > cell_limit).float() * (reminder-cell_limit)
                                    rram_map2_sml = (reminder <= cell_limit).float() * 0
                                    rram_map[(numColPerSynapse)*(j*2) + k] = rram_map1_big + rram_map1_sml
                                    rram_map[(numColPerSynapse)*(j*2) + (k+1)] = rram_map2_big + rram_map2_sml
                        w_prune = torch.zeros(tmp_w.shape)
                        if n_q == 0:
                            for j in range(0,rram_map.shape[0],(numColPerSynapse)):
                                ### this is for weightbit = 8 cellbit =4
                                if numColPerSynapse == 2:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) * (2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-level-1)).float() * rram_map[j+1] * (2**CellBit)

                                ### this is for weightbit =8 cellbit = 2
                                if numColPerSynapse == 4:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float()*(2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit) + (rram_map[j+2]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)*(2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float()*(rram_map[j]) + (rram_map[j+1]<=(2**CellBit-level-1)).float()*(rram_map[j+1])*(2**CellBit) + (rram_map[j+2]<=(2**CellBit-level-1)).float()*(rram_map[j+2])*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]<=(2**CellBit-level-1)).float()*(rram_map[j+3])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                                if numColPerSynapse == 8:
                                    resumedata_prune = 0
                                    resumedata_noprune = rram_map[j] + rram_map[j+1]*(2**CellBit) + rram_map[j+2]*(2**CellBit)*(2**CellBit) + rram_map[j+3]*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+4]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+5]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+6]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+7]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)



                                resumedata = resumedata_prune + resumedata_noprune
                                w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune

                        if n_q == 1:
                            for j in range(0,rram_map.shape[0], numColPerSynapse*2):
                                if numColPerSynapse == 2:
                                    #print('j=',j)
                                    #print('rram_map size = ', rram_map.shape)
                                    resumedata = (rram_map[j]+rram_map[j+1]) + (rram_map[j+2]+rram_map[j+3])*(2**CellBit)

                                if numColPerSynapse == 4:
                                    resumedata = (rram_map[j]+rram_map[j+1]) + (rram_map[j+2]+rram_map[j+3])*(2**CellBit) + (rram_map[j+3]+rram_map[j+4])*(2**CellBit)*(2**CellBit) + (rram_map[j+5]+rram_map[j+6])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                                if numColPerSynapse == 8:
                                    print('8 bit weight with 1 bit cell resolution can not use nonuniform quantization !!')



                                w_prune[int(j/(numColPerSynapse*2))] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax
                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune



                param.data = w.reshape(param.data.shape)


            if name == 'features.module.12.weight':




                ####### subarray heat distribution1 #####
                #different_level = 4
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 9
                #        level = 4
                #    if indx == 1:
                #        front = 9
                #        end = 17
                #        level = 3
                #    if indx == 2:
                #        front = 17
                #        end = 45
                #        level = 2
                #    if indx == 3:
                #        front = 45
                #        end = 61
                #        level = 1
                #    if indx == 4:
                #        front = 61
                #        end = 144
                #        level = 0
                #############################################

                ###### subarray heat distribution1 prune and retrain #####
                #different_level = 2
                #front=0
                #end=0
                #n_q = 0
                #cell_limit = 7
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 72
                #        level = 0
                #        n_q = 0
                #    if indx == 1:
                #        front = 72
                #        end = 144
                #        level = 4
                #        n_q = 0
                ############################################

                ###### subarray heat distribution1 prune and retrain 4x4 sliding window quantization#####
                different_level = 3
                front=0
                end=0
                n_q = 0
                cell_limit = 7
                for indx in range(different_level):
                    if indx == 0:
                        front = 0
                        end = 72
                        level = 0
                        n_q = 0
                    if indx == 1:
                        front = 72
                        end = 82
                        level = 0
                        n_q = 1
                    if indx == 2:
                        front = 82
                        end = 144
                        level = 0
                        n_q = 0
                ############################################



                ###### subarray heat distribution2 ###########################
                #different_level = 5
                #front=0
                #end=0
                #n_q = 0
                #cell_limit = 7
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 20
                #        level = 4
                #        n_q = 0
                #    if indx == 1:
                #        front =20
                #        end = 24
                #        level = 3
                #        n_q = 0
                #    if indx == 2:
                #        front = 24
                #        end = 37
                #        level = 2
                #        n_q = 0
                #    if indx == 3:
                #        front = 37
                #        end = 54
                #        level = 1
                #        n_q = 0
                #    if indx == 4:
                #        front = 54
                #        end = 144
                #        level = 0
                #        n_q = 0
                #####################################################################

                ####### subarray heat distribution2 after pruning and retrain#######
                #different_level = 2
                #front=0
                #end=0
                #n_q=0
                #cell_limit= 7
                #for indx in range(different_level):
                #    if indx==0:
                #        front= 0
                #        end= 72
                #        level= 0
                #        n_q= 0
                #    if indx==1:
                #        front= 72
                #        end= 144
                #        level= 0
                #        n_q= 0
                ###################################################################






                    for i in range(front,end):
                        key,value = rram_partition_total_sum_list[i]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        #print('key = {} , value = {}'.format(key, value))

                        if n_q == 0:
                            rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse), int(tmp_w.shape[1]))
                        if n_q == 1:
                            rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse)*2, int(tmp_w.shape[1]))
                        #rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            ## normalized data
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            if n_q == 0 :
                                for k in range(numColPerSynapse):
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                                    rram_map[(numColPerSynapse)*j+k] = reminder
                            if n_q == 1 :
                                for k in range(0,2*numColPerSynapse,2):
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange)
                                    rram_map1_big = (reminder > cell_limit).float() * cell_limit
                                    rram_map1_sml = (reminder <= cell_limit).float() * reminder
                                    rram_map2_big = (reminder > cell_limit).float() * (reminder-cell_limit)
                                    rram_map2_sml = (reminder <= cell_limit).float() * 0
                                    rram_map[(numColPerSynapse)*(j*2) + k] = rram_map1_big + rram_map1_sml
                                    rram_map[(numColPerSynapse)*(j*2) + (k+1)] = rram_map2_big + rram_map2_sml
                        w_prune = torch.zeros(tmp_w.shape)
                        if n_q == 0:
                            for j in range(0,rram_map.shape[0],(numColPerSynapse)):
                                ### this is for weightbit = 8 cellbit =4
                                if numColPerSynapse == 2:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) * (2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-level-1)).float() * rram_map[j+1] * (2**CellBit)

                                ### this is for weightbit =8 cellbit = 2
                                if numColPerSynapse == 4:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float()*(2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit) + (rram_map[j+2]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)*(2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float()*(rram_map[j]) + (rram_map[j+1]<=(2**CellBit-level-1)).float()*(rram_map[j+1])*(2**CellBit) + (rram_map[j+2]<=(2**CellBit-level-1)).float()*(rram_map[j+2])*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]<=(2**CellBit-level-1)).float()*(rram_map[j+3])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                                if numColPerSynapse == 8:
                                    resumedata_prune = 0
                                    resumedata_noprune = rram_map[j] + rram_map[j+1]*(2**CellBit) + rram_map[j+2]*(2**CellBit)*(2**CellBit) + rram_map[j+3]*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+4]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+5]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+6]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+7]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)



                                resumedata = resumedata_prune + resumedata_noprune
                                w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune

                        if n_q == 1:
                            for j in range(0,rram_map.shape[0], numColPerSynapse*2):
                                if numColPerSynapse == 2:
                                    #print('j=',j)
                                    #print('rram_map size = ', rram_map.shape)
                                    resumedata = (rram_map[j]+rram_map[j+1]) + (rram_map[j+2]+rram_map[j+3])*(2**CellBit)

                                if numColPerSynapse == 4:
                                    resumedata = (rram_map[j]+rram_map[j+1]) + (rram_map[j+2]+rram_map[j+3])*(2**CellBit) + (rram_map[j+3]+rram_map[j+4])*(2**CellBit)*(2**CellBit) + (rram_map[j+5]+rram_map[j+6])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                                if numColPerSynapse == 8:
                                    print('8 bit weight with 1 bit cell resolution can not use nonuniform quantization !!')



                                w_prune[int(j/(numColPerSynapse*2))] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune



                param.data = w.reshape(param.data.shape)





        if len(param.data.size()) == 2 :
            row = param.data.shape[1]
            col = param.data.shape[0]*numColPerSynapse
            w = param.data.reshape(param.data.shape[0],row)

            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_row_sum = {} ## a row sum in subarray
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])
            #rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1],reverse=True)

            if name == 'classifier.0.weight':



                ######### subarray heat distribution 1 #######
                #different_level = 5
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 195
                #        level = 4
                #    if indx == 1:
                #        front = 195
                #        end = 273
                #        level = 3
                #    if indx == 2:
                #        front = 273
                #        end = 329
                #        level = 2
                #    if indx == 3:
                #        front = 329
                #        end = 390
                #        level = 1
                #    if indx == 4:
                #        front = 390
                #        end = 512
                #        level = 0
                # ############################################


                ##### subarray heat distribution 1 prune and retrain#####
                #different_level = 5
                #front=0
                #end=0
                #n_q = 0
                #cell_limit = 7
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 256
                #        level = 0
                #        n_q = 0
                #    if indx == 1:
                #        front = 256
                #        end = 271
                #        level = 7
                #        n_q = 1
                #    if indx == 2:
                #        front = 271
                #        end = 334
                #        level = 6
                #        n_q = 1
                #    if indx == 3:
                #        front = 334
                #        end = 396
                #        level = 5
                #        n_q = 1
                #    if indx == 4:
                #        front = 396
                #        end = 512
                #        level = 4
                #        n_q = 0
                #################################################


                ######## subarray heat distribution 1 prune and retrain 4x4 sliding window quantization#######
                different_level = 9
                front=0
                end=0
                n_q = 0
                cell_limit = 7
                for indx in range(different_level):
                    if indx == 0:
                        front = 0
                        end = 256
                        level = 0
                        n_q = 0
                    if indx == 1:
                        front = 256
                        end = 266
                        level = 3
                        n_q = 1
                    if indx == 2:
                        front = 266
                        end = 271
                        level = 3
                        n_q = 1
                    if indx == 3:
                        front = 271
                        end = 294
                        level = 2
                        n_q = 1
                    if indx == 4:
                        front = 294
                        end = 334
                        level = 2
                        n_q = 1
                    if indx == 5:
                        front = 334
                        end = 339
                        level = 1
                        n_q = 1
                    if indx == 6:
                        front = 339
                        end = 396
                        level = 1
                        n_q = 1
                    if indx == 7:
                        front = 396
                        end = 399
                        level = 0
                        n_q = 1
                    if indx == 8:
                        front = 399
                        end = 512
                        level = 0
                        n_q = 0
                ################################################




                ########## sbuarray heat distribution 2 ##########33
                #different_level = 5
                #front=0
                #end=0
                #n_q = 0
                #cell_limit = 7

                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 118
                #        level = 4
                #        n_q = 0
                #    if indx == 1:
                #        front = 118
                #        end = 188
                #        level = 3
                #        n_q = 0
                #    if indx == 2:
                #        front = 188
                #        end = 328
                #        level = 2
                #        n_q = 0
                #    if indx == 3:
                #        front = 328
                #        end = 414
                #        level = 1
                #        n_q = 0
                #    if indx == 4:
                #        front = 414
                #        end = 512
                #        level = 0
                #        n_q = 0

                #################################################

                ####### subarray heat distribution2 after prune and retrain #####
                #different_level =4
                #front=0
                #end=0
                #n_q=0
                #cell_limit = 7
                #for indx in range(different_level):
                #    if indx==0:
                #        front= 0
                #        end= 256
                #        level= 0
                #        n_q= 0
                #    if indx==1:
                #        front= 256
                #        end= 332
                #        level= 2
                #        n_q= 0
                #    if indx==2:
                #        front= 332
                #        end= 419
                #        level= 1
                #        n_q= 0
                #    if indx==3:
                #        front= 419
                #        end= 512
                #        level= 0
                #        n_q= 0
                #################################################################




                    for i in range(front,end):
                        key,value = rram_partition_total_sum_list[i]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]


                        if n_q == 0:
                            rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse), int(tmp_w.shape[1]))
                        if n_q == 1:
                            rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse)*2, int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            ## normalized data
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            if n_q == 0 :
                                for k in range(numColPerSynapse):
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                                    rram_map[(numColPerSynapse)*j+k] = reminder
                            if n_q == 1 :
                                for k in range(0,2*numColPerSynapse,2):
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange)
                                    rram_map1_big = (reminder > cell_limit).float() * cell_limit
                                    rram_map1_sml = (reminder <= cell_limit).float() * reminder
                                    rram_map2_big = (reminder > cell_limit).float() * (reminder-cell_limit)
                                    rram_map2_sml = (reminder <= cell_limit).float() * 0
                                    rram_map[(numColPerSynapse)*(j*2) + k] = rram_map1_big + rram_map1_sml
                                    rram_map[(numColPerSynapse)*(j*2) + (k+1)] = rram_map2_big + rram_map2_sml


                        w_prune = torch.zeros(tmp_w.shape)
                        if n_q == 0:
                            for j in range(0,rram_map.shape[0],(numColPerSynapse)):
                                ### this is for weightbit = 8 cellbit =4
                                if numColPerSynapse == 2:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) * (2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-level-1)).float() * rram_map[j+1] * (2**CellBit)

                                ### this is for weightbit =8 cellbit = 2
                                if numColPerSynapse == 4:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float()*(2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit) + (rram_map[j+2]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)*(2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float()*(rram_map[j]) + (rram_map[j+1]<=(2**CellBit-level-1)).float()*(rram_map[j+1])*(2**CellBit) + (rram_map[j+2]<=(2**CellBit-level-1)).float()*(rram_map[j+2])*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]<=(2**CellBit-level-1)).float()*(rram_map[j+3])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                                if numColPerSynapse == 8:
                                    resumedata_prune = 0
                                    resumedata_noprune = rram_map[j] + rram_map[j+1]*(2**CellBit) + rram_map[j+2]*(2**CellBit)*(2**CellBit) + rram_map[j+3]*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+4]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+5]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+6]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+7]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)



                                resumedata = resumedata_prune + resumedata_noprune
                                w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune

                        if n_q == 1:
                            for j in range(0,rram_map.shape[0], numColPerSynapse*2):
                                if numColPerSynapse == 2:
                                    resumedata = (rram_map[j]+rram_map[j+1]) + (rram_map[j+2]+rram_map[j+3])*(2**CellBit)

                                if numColPerSynapse == 4:
                                    resumedata = (rram_map[j]+rram_map[j+1]) + (rram_map[j+2]+rram_map[j+3])*(2**CellBit) + (rram_map[j+3]+rram_map[j+4])*(2**CellBit)*(2**CellBit) + (rram_map[j+5]+rram_map[j+6])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                                if numColPerSynapse == 8:
                                    print('8 bit weight with 1 bit cell resolution can not use nonuniform quantization !!')



                                w_prune[int(j/(numColPerSynapse*2))] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax
                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune

                param.data = w.reshape(param.data.shape)


            if name == 'classifier.2.weight':


                ######### subarray heat distribution 1 ####################
                #different_level = 2
                #front=0
                #end=0
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 3
                #        level = 2
                #    if indx == 1:
                #        front = 3
                #        end = 8
                #        level = 1
                ###########################################################

                ######### subarray heat distribution 1 prune and retrain####################
                #different_level = 2
                #front=0
                #end=0
                #n_q = 0
                #cell_limit = 7
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 4
                #        level = 0
                #        n_q = 0
                #    if indx == 1:
                #        front = 4
                #        end = 8
                #        level = 5
                #        n_q = 1
                ###########################################################


                ######## subarray heat distribution 1 prune and retrain 4x4 sliding window quantization####################
                different_level = 2
                front=0
                end=0
                n_q = 0
                cell_limit = 7
                for indx in range(different_level):
                    if indx == 0:
                        front = 0
                        end = 4
                        level = 0
                        n_q = 0
                    if indx == 1:
                        front = 4
                        end = 8
                        level = 1
                        n_q = 1
                ##########################################################



                ######## subaray heat distribution 2###################
                #different_level = 2
                #front=0
                #end=0
                #n_q = 0
                #cell_limit = 7
                #for indx in range(different_level):
                #    if indx == 0:
                #        front = 0
                #        end = 5
                #        level = 4
                #        n_q = 0

                #    if indx == 1:
                #        front = 5
                #        end = 8
                #        level = 3
                #        n_q = 0
                ############################################################

                ######## subarray heat distribution 2 after prune and retrain##???
                #different_level =3
                #front= 0
                #end=0
                #n_q=1
                #cell_limit= 7
                #for indx in range(different_level):
                #    if indx==0:
                #        front= 0
                #        end= 4
                #        level= 0
                #        n_q= 0
                #    if indx==1:
                #        front= 4
                #        end= 5
                #        level= 4
                #        n_q= 0
                #    if indx==2:
                #        front= 5
                #        end= 8
                #        level= 3
                #        n_q= 0

                ##############################################################




                    for i in range(front,end):
                        key,value = rram_partition_total_sum_list[i]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]


                        if n_q == 0:
                            rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse), int(tmp_w.shape[1]))
                        if n_q == 1:
                            rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse)*2, int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            ## normalized data
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            if n_q == 0 :
                                for k in range(numColPerSynapse):
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                                    rram_map[(numColPerSynapse)*j+k] = reminder
                            if n_q == 1 :
                                for k in range(0,2*numColPerSynapse,2):
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange)
                                    rram_map1_big = (reminder > cell_limit).float() * cell_limit
                                    rram_map1_sml = (reminder <= cell_limit).float() * reminder
                                    rram_map2_big = (reminder > cell_limit).float() * (reminder-cell_limit)
                                    rram_map2_sml = (reminder <= cell_limit).float() * 0
                                    rram_map[(numColPerSynapse)*(j*2) + k] = rram_map1_big + rram_map1_sml
                                    rram_map[(numColPerSynapse)*(j*2) + (k+1)] = rram_map2_big + rram_map2_sml


                        w_prune = torch.zeros(tmp_w.shape)
                        if n_q == 0:
                            for j in range(0,rram_map.shape[0],(numColPerSynapse)):
                                ### this is for weightbit = 8 cellbit =4
                                if numColPerSynapse == 2:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float() * (2**CellBit-level-1) * (2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-level-1)).float() * rram_map[j+1] * (2**CellBit)

                                ### this is for weightbit =8 cellbit = 2
                                if numColPerSynapse == 4:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-level-1)).float()*(2**CellBit-level-1) + (rram_map[j+1]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit) + (rram_map[j+2]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]>(2**CellBit-level-1)).float()*(2**CellBit-level-1)*(2**CellBit)*(2**CellBit)*(2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-level-1)).float()*(rram_map[j]) + (rram_map[j+1]<=(2**CellBit-level-1)).float()*(rram_map[j+1])*(2**CellBit) + (rram_map[j+2]<=(2**CellBit-level-1)).float()*(rram_map[j+2])*(2**CellBit)*(2**CellBit)  + (rram_map[j+3]<=(2**CellBit-level-1)).float()*(rram_map[j+3])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                                if numColPerSynapse == 8:
                                    resumedata_prune = 0
                                    resumedata_noprune = rram_map[j] + rram_map[j+1]*(2**CellBit) + rram_map[j+2]*(2**CellBit)*(2**CellBit) + rram_map[j+3]*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+4]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+5]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+6]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit) + rram_map[j+7]*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)*(2**CellBit)



                                resumedata = resumedata_prune + resumedata_noprune
                                w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune

                        if n_q == 1:
                            for j in range(0,rram_map.shape[0], numColPerSynapse*2):
                                if numColPerSynapse == 2:
                                    resumedata = (rram_map[j]+rram_map[j+1]) + (rram_map[j+2]+rram_map[j+3])*(2**CellBit)

                                if numColPerSynapse == 4:
                                    resumedata = (rram_map[j]+rram_map[j+1]) + (rram_map[j+2]+rram_map[j+3])*(2**CellBit) + (rram_map[j+3]+rram_map[j+4])*(2**CellBit)*(2**CellBit) + (rram_map[j+5]+rram_map[j+6])*(2**CellBit)*(2**CellBit)*(2**CellBit)

                                if numColPerSynapse == 8:
                                    print('8 bit weight with 1 bit cell resolution can not use nonuniform quantization !!')



                                w_prune[int(j/(numColPerSynapse*2))] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax
                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune

                param.data = w.reshape(param.data.shape)




#def prune_layer(_model,args, chromosome, size): ### for GA use
#def prune_layer(_model, args, prune_ratio_list): ### for prune_layer only
def prune_layer(_model,args):

    realmin=1000
    realmax=-9999
    for name, param in (_model.named_parameters()):
        if len(param.data.size()) != 1:
            w = param.data

            if torch.min(param.data)<realmin:
                realmin = torch.min(param.data)
            if torch.max(param.data)>realmax:
                realmax = torch.max(param.data)

    #print('realmin = ', realmin)
    #print('realmax = ', realmax)



    SynapseBit = 8
    CellBit = 4
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = realmin
    RealMax = realmax
    SubarrayRow = 128
    SubarrayCol = 128


    len_ts = 0
    total_pruned_cell = 0
    total_subarray_cell = 0
    all_list= []
    for name, param in (_model.named_parameters()):
        ## map convolutional layer into rram crossbar with 8bit weight and 4bit cellbit
        if len(param.data.size()) == 4 :


            row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            #col = param.data.shape[0]*numColPerSynapse
            w = param.data.reshape(param.data.shape[0],row)
            #rram_map = torch.zeros(int(col),int(row))

            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_row_sum = {} ## a row sum in subarray
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                all_tmp = [key, value, name]
                rram_partition_total_sum_list.append(tmp)
                all_list.append(all_tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])
            #rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1], reverse = True)

            len_ts += len(rram_partition_total_sum)


            ############## for vgg8 #####################
            if args.arch == 'vgg8':
                #print('name = ', name)
                #ratio=int(args.prune_ratio)/100
                if name == 'module.features.module.2.weight':
                    #print('layer 2 prune ratio = ', chromosome[1])
                    #ratio = chromosome[1]/100
                    ratio = prune_ratio_list[1]
                    #ratio = 0.1
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    #print('layer1 pruned percentage =  0.0')
                    #print('layer2 have ',len(rram_partition_total_sum),' subarrays')
                    #print('layer2 pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(128,128)
                    #print('layer2 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol


                if name == 'module.features.module.5.weight':
                    #print('layer 3 prune ratio = ', chromosome[2])
                    #ratio = chromosome[2]/100
                    ratio = prune_ratio_list[2]
                    #ratio = 0.2
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    #print('layer3 have ',len(rram_partition_total_sum),' subarrays')
                    #print('layer3 pruned ',pruned_subarray,' subarrays. ')

                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(128,128)

                    #print('layer3 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol


                if name == 'module.features.module.7.weight':
                    #print('layer 4 prune ratio = ', chromosome[3])
                    #ratio = chromosome[3]/100
                    ratio = prune_ratio_list[3]
                    #ratio = 0.3
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    #print('layer4 have ',len(rram_partition_total_sum),' subarrays')
                    #print('layer4 pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(128,128)
                    #print('layer4 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol

                if name == 'module.features.module.10.weight':
                    #print('layer 5 prune ratio = ', chromosome[4])
                    #ratio = chromosome[4]/100
                    ratio = prune_ratio_list[4]
                    #ratio = 0.3
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    #print('layer5 have ',len(rram_partition_total_sum),' subarrays')
                    #print('layer5 pruned ',pruned_subarray,' subarrays. ')

                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(128,128)
                    #print('layer5 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol

                if name == 'module.features.module.12.weight':
                    #print('layer 6 prune ratio = ', chromosome[5])
                    #ratio = chromosome[5]/100
                    ratio = prune_ratio_list[5]
                    #ratio = 0.3
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    #print('layer6 have ',len(rram_partition_total_sum),' subarrays')
                    #print('layer6 pruned ',pruned_subarray,' subarrays. ')

                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(128,128)
                    #print('layer6 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol


                param.data = w.reshape(param.data.shape)


            if args.arch == 'vgg11':
                ratio= int(args.prune_ratio)/100
                if name == 'module.features.module.3.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer1 pruned percentage =  0.0')
                    print('layer2 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer2 pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                    print('layer2 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol
                if name == 'module.features.module.6.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer3 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer3 pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                    print('layer3 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol
                if name == 'module.features.module.8.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer4 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer4 pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                    print('layer4 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol
                if name == 'module.features.module.11.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer5 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer5 pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                    print('layer5 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol
                if name == 'module.features.module.13.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer6 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer6 pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                    print('layer6 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol
                if name == 'module.features.module.16.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer7 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer7 pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                    print('layer7 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol
                if name == 'module.features.module.18.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer8 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer8 pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                    print('layer8 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol


                param.data = w.reshape(param.data.shape)



            if args.arch == 'alexnet':
                ratio = int(args.prune_ratio)/100
                #if name == 'module.features.0.weight':
                #    #ratio = 0.6
                #    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                #    print('layer1 have ',len(rram_partition_total_sum),' subarrays')
                #    print('layer1 pruned ',pruned_subarray,' subarrays. ')
                #    for i in range(pruned_subarray):
                #        key,value = rram_partition_total_sum_list[i]
                #        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                #        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                #    print('layer1 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                #    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol
                if name == 'module.features.3.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer2 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer2 pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                    print('layer2 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol
                if name == 'module.features.6.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer3 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer3 pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                    print('layer3 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol
                if name == 'module.features.8.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer4 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer4 pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                    print('layer4 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol
                if name == 'module.features.10.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer5 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer5 pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                    print('layer5 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol

                param.data = w.reshape(param.data.shape)

            if args.arch == 'resnet34':
                ratio = int(args.prune_ratio)/100
                print(name)
                if name == 'module.layer1.0.conv1.weight' or name =='module.layer1.0.conv2.weight' or name =='module.layer1.1.conv1.weight' or name =='module.layer1.1.conv2.weight' or name == 'module.layer1.2.conv1.weight' or name == 'module.layer1.2.conv2.weight' or name =='module.layer2.0.conv1.weight' or name == 'module.layer2.0.conv2.weight' or name =='module.layer2.1.conv1.weight' or name == 'module.layer2.1.conv2.weight' or name == 'module.layer2.2.conv1.weight' or name == 'module.layer2.2.conv2.weight' or name == 'module.layer2.3.conv1.weight' or name == 'module.layer2.3.conv2.weight' or name == 'module.layer3.0.conv1.weight' or name == 'module.layer3.0.conv2.weight' or name == 'module.layer3.0.shortcut.0.weight' or name == 'module.layer3.1.conv1.weight' or name == 'module.layer3.1.conv2.weight' or name == 'module.layer3.2.conv1.weight' or name == 'module.layer3.2.conv2.weight' or name == 'module.layer3.3.conv1.weight' or name =='module.layer3.3.conv2.weight' or name == 'module.layer3.4.conv1.weight' or name == 'module.layer3.4.conv2.weight' or name == 'module.layer3.5.conv1.weight' or name =='module.layer3.5.conv2.weight' or name == 'module.layer4.0.conv1.weight' or name == 'module.layer4.0.conv2.weight' or name == 'module.layer4.0.shortcut.0.weight' or name == 'module.layer4.1.conv1.weight' or name == 'module.layer4.1.conv2.weight' or name == 'module.layer4.2.conv1.weight' or name == 'module.layer4.2.conv2.weight':
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print(name, ' have ',len(rram_partition_total_sum),' subarrays')
                    print(name, ' pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                    print(name,' pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol

                param.data = w.reshape(param.data.shape)






        if len(param.data.size()) == 2:


            row = param.data.shape[1]
            col = param.data.shape[0]*numColPerSynapse
            w = param.data.reshape(param.data.shape[0],row)
            rram_map = torch.zeros(int(col),int(row))


            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_row_sum = {}
            rram_partition_total_sum = {}
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_row_sum[i,j] = torch.sum(rram_partition[i,j], dim=1)
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
                    #rram_partition_total_sum[i,j] = torch.sum(rram_partition[i,j])

            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                all_tmp = [key, value, name]
                rram_partition_total_sum_list.append(tmp)
                all_list.append(all_tmp)


            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])
            #rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1], reverse = True)
            len_ts += len(rram_partition_total_sum)

            #### for vgg8 #####
            if args.arch == 'vgg8':
                if name == 'module.classifier.0.weight':
                    #print('layer 7 prune ratio = ', chromosome[6])
                    #ratio = chromosome[6]/100
                    ratio = prune_ratio_list[6]
                    #ratio = 0.9
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    #print('layer7 have ',len(rram_partition_total_sum),' subarrays')
                    #print('layer7 pruned ',pruned_subarray,' subarrays. ')

                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(128,128)

                        for k in range(len(all_list)):
                            if all_list[k][2]=='classifier.0.weight':
                                if all_list[k][0] == key:
                                    all_list[k].pop(1)
                                    all_list[k].insert(1,torch.tensor(-999))


                    #print('layer7 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol


                if name == 'module.classifier.2.weight':
                    #print('layer8 prune_ratio = ', chromosome[7])
                    #ratio = chromosome[7]/100
                    ratio = prune_ratio_list[7]
                    #ratio = 0
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    #print('layer8 have ',len(rram_partition_total_sum),' subarrays')
                    #print('layer8 pruned ',pruned_subarray,' subarrays. ')

                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(10,128)
                        for k in range(len(all_list)):
                            if all_list[k][2]=='classifier.2.weight':
                                if all_list[k][0] == key:
                                    all_list[k].pop(1)
                                    all_list[k].insert(1,torch.tensor(-999))


                    #print('layer8 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol

                param.data = w.reshape(param.data.shape)

            ##### for vgg11 ###
            if args.arch == 'vgg11':
                ratio= int(args.prune_ratio)/100
                if name == 'module.classifier.0.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer9 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer9 pruned ',pruned_subarray,' subarrays. ')

                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)

                        for k in range(len(all_list)):
                            if all_list[k][2]=='module.classifier.0.weight':
                                if all_list[k][0] == key:
                                    all_list[k].pop(1)
                                    all_list[k].insert(1,torch.tensor(-999))


                    print('layer7 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol
                if name == 'module.classifier.2.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer10 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer10 pruned ',pruned_subarray,' subarrays. ')
                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)

                        for k in range(len(all_list)):
                            if all_list[k][2]=='module.classifier.2.weight':
                                if all_list[k][0] == key:
                                    all_list[k].pop(1)
                                    all_list[k].insert(1,torch.tensor(-999))
                    print('layer7 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol
                if name == 'module.classifier.4.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer11 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer11 pruned ',pruned_subarray,' subarrays. ')

                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)

                        for k in range(len(all_list)):
                            if all_list[k][2]=='module.classifier.4.weight':
                                if all_list[k][0] == key:
                                    all_list[k].pop(1)
                                    all_list[k].insert(1,torch.tensor(-999))


                    print('layer11 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))
                    total_pruned_cell += pruned_subarray*SubarrayRow*SubarrayCol


                param.data = w.reshape(param.data.shape)

            if args.arch == 'alexnet':
                ratio = int(args.prune_ratio)/100
                if name == 'module.classifier.0.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer6 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer6 pruned ',pruned_subarray,' subarrays. ')

                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)

                        for k in range(len(all_list)):
                            if all_list[k][2]=='module.classifier.0.weight':
                                if all_list[k][0] == key:
                                    all_list[k].pop(1)
                                    all_list[k].insert(1,torch.tensor(-999))

                    print('layer6 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))

                if name == 'module.classifier.2.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer7 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer7 pruned ',pruned_subarray,' subarrays. ')

                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)

                        for k in range(len(all_list)):
                            if all_list[k][2]=='module.classifier.2.weight':
                                if all_list[k][0] == key:
                                    all_list[k].pop(1)
                                    all_list[k].insert(1,torch.tensor(-999))

                    print('layer7 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))

                if name == 'module.classifier.4.weight':
                    #ratio = 0.6
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer8 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer8 pruned ',pruned_subarray,' subarrays. ')

                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)

                        for k in range(len(all_list)):
                            if all_list[k][2]=='module.classifier.4.weight':
                                if all_list[k][0] == key:
                                    all_list[k].pop(1)
                                    all_list[k].insert(1,torch.tensor(-999))

                    print('layer8 pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))

                param.data = w.reshape(param.data.shape)
            if args.arch =='resnet34':
                ratio = int(args.prune_ratio)/100
                print(name)
                if name == 'module.linear.weight':
                    pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print(name, ' have ',len(rram_partition_total_sum),' subarrays')
                    print(name, ' pruned ',pruned_subarray,' subarrays. ')

                    for i in range(pruned_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        a = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)

                        for k in range(len(all_list)):
                            if all_list[k][2]=='module.linear.weight':
                                if all_list[k][0] == key:
                                    all_list[k].pop(1)
                                    all_list[k].insert(1,torch.tensor(-999))

                    print(name, ' pruned percentage = ', pruned_subarray/len(rram_partition_total_sum))

                param.data = w.reshape(param.data.shape)






    all_list_sorted = sorted(all_list, key = lambda s:s[1])

    total_subarray_cell = len_ts*SubarrayRow*SubarrayCol
    #print('total_pruned_cell = ', total_pruned_cell)
    #print('total_subarray_cell = ', total_subarray_cell)
    #print('total_pruned percentage = ', total_pruned_cell/total_subarray_cell)

    #print('all list after sorted = ', all_list_sorted)

    total_prune_ratio = total_pruned_cell/total_subarray_cell
    return  total_prune_ratio


def downgrading(model, args): ## this function assume one 8bit weight are mapped to 2 4bit cell
    realmin=1000
    realmax=-9999
    for name, param in (model.named_parameters()):
        if len(param.data.size()) != 1:

            if torch.min(param.data)<realmin:
                realmin = torch.min(param.data)
            if torch.max(param.data)>realmax:
                realmax = torch.max(param.data)
    print('realmin = ', realmin)
    print('realmax = ', realmax)
    SynapseBit = 8
    CellBit = 4
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = realmin
    RealMax = realmax
    SubarrayRow = 128
    SubarrayCol = 128
    for name, param in (model.named_parameters()):
        if len(param.data.size()) == 4 :
            print('name=',name)
            row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))

            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])
            for i in range(len(rram_partition_total_sum_list)):
                key, value = rram_partition_total_sum_list[i]
                tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                for j in range(tmp_w.shape[0]):
                    newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                    value = newdata
                    cellrange = 2**CellBit
                    for k in range(numColPerSynapse):
                        reminder = torch.ceil(value%cellrange)
                        value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                        rram_map[(numColPerSynapse)*j+k] = reminder
                w_BD = torch.zeros(tmp_w.shape)
                for j in range(0,rram_map.shape[0],(numColPerSynapse)):
                    for k in range(len(rram_map[j])):
                        tmp_bin1 = int(rram_map[j][k])
                        tmp_bin2 = int(rram_map[j+1][k])
                        LSB1 = tmp_bin1&1
                        LSB2 = tmp_bin2&1
                        tmp_bin1 = tmp_bin1 - LSB1
                        tmp_bin2 = tmp_bin2 - LSB2
                        rram_map[j][k] = tmp_bin1
                        rram_map[j+1][k] = tmp_bin2
                        tmp_bin = tmp_bin1*(2**CellBit) + tmp_bin2
                    w_BD[int(j/numColPerSynapse)] = ((tmp_bin-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_BD
            param.data = w.reshape(param.data.shape)
        if len(param.data.size()) == 2 :
            print('name=',name)
            row = param.data.shape[1]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)
            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])
            for i in range(len(rram_partition_total_sum_list)):
                key, value = rram_partition_total_sum_list[i]
                tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                for j in range(tmp_w.shape[0]):
                    newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                    value = newdata
                    cellrange = 2**CellBit
                    for k in range(numColPerSynapse):
                        reminder = torch.ceil(value%cellrange)
                        value = torch.floor(value/cellrange) ## we use floor here instead of ceil in original Chip.cpp
                        rram_map[(numColPerSynapse)*j+k] = reminder
                w_BD = torch.zeros(tmp_w.shape)
                for j in range(0,rram_map.shape[0],(numColPerSynapse)):
                    for k in range(len(rram_map[j])):
                        tmp_bin1 = int(rram_map[j][k])
                        tmp_bin2 = int(rram_map[j+1][k])
                        LSB1 = tmp_bin1&1
                        LSB2 = tmp_bin2&1
                        tmp_bin1 = tmp_bin1 - LSB1
                        tmp_bin2 = tmp_bin2 - LSB2
                        rram_map[j][k] = tmp_bin1
                        rram_map[j+1][k] = tmp_bin2
                        tmp_bin = tmp_bin1*(2**CellBit) + tmp_bin2
                    w_BD[int(j/numColPerSynapse)] = ((tmp_bin-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune
            param.data = w.reshape(param.data.shape)

def downgrading_8bit(model, args, sector, idx): ### this function assume a weight mapped to one cell
    realmin=1000
    realmax=-9999
    for name, param in (model.named_parameters()):
        if len(param.data.size()) != 1:

            if torch.min(param.data)<realmin:
                realmin = torch.min(param.data)
            if torch.max(param.data)>realmax:
                realmax = torch.max(param.data)
    print('realmin = ', realmin)
    print('realmax = ', realmax)
    SynapseBit = args.testbit ####?????????????????????normalizedmax min??????
    CellBit = args.testbit
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = realmin
    RealMax = realmax
    SubarrayRow = 128
    SubarrayCol = 128
    for name, param in (model.named_parameters()):
        if len(param.data.size()) == 4 :
            print('name=',name)
            row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))

            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])
            for s in range(idx):
                if sector[s][4]==name:
                    key = sector[s][2]
                    if int(sector[s][1])==5 or int(sector[s][1])==6  :
                        print('s=',s)
                        tmp_w_conv = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        rram_map_conv = torch.zeros(int(tmp_w_conv.shape[0]),int(tmp_w_conv.shape[1]))
                        for j in range(tmp_w_conv.shape[0]):
                            newdata_conv = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w_conv[j]-RealMax) + NormalizedMax).round()
                            rram_map_conv[j] = newdata_conv
                        w_BD_conv = torch.zeros(tmp_w_conv.shape)
                        for j in range(rram_map_conv.shape[0]):
                            for k in range(len(rram_map_conv[j])):
                                tmp_bin_conv = int(rram_map_conv[j][k])
                                LSB_conv = tmp_bin_conv&1
                                tmp_bin_conv = tmp_bin_conv - LSB_conv
                                rram_map_conv[j][k] = tmp_bin_conv
                            w_BD_conv[j] = ((rram_map_conv[j]-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_BD_conv
                    else:
                        heat_value = (int(sector[s][1])/16)*(2**SynapseBit)
                        print('origitnal heat = ', sector[s][1])
                        print('heat value = ', heat_value)
                        tmp_w2 = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        rram_map = torch.zeros(int(tmp_w2.shape[0]),int(tmp_w2.shape[1]))
                        for j in range(tmp_w2.shape[0]):
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w2[j]-RealMax) + NormalizedMax).round()
                            D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                            resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                            tmp_w2[j]=resumedata
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w2
            param.data = w.reshape(param.data.shape)
        if len(param.data.size()) == 2 :
            print('name=',name)
            row = param.data.shape[1]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)
            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])
            for s in range(idx):
                if sector[s][4]==name:
                    key = sector[s][2]
                    if int(sector[s][1])==5 or int(sector[s][1])==6 :
                        print('s=',s)
                        tmp_w_fc = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        rram_map_fc = torch.zeros(int(tmp_w_fc.shape[0]),int(tmp_w_fc.shape[1]))
                        for j in range(tmp_w_fc.shape[0]):
                            newdata_fc = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w_fc[j]-RealMax) + NormalizedMax).round()
                            rram_map_fc[j] = newdata_fc
                        w_BD_fc = torch.zeros(tmp_w_fc.shape)
                        for j in range(rram_map_fc.shape[0]):
                            for k in range(len(rram_map_fc[j])):
                                tmp_bin_fc = int(rram_map_fc[j][k])
                                LSB_fc = tmp_bin_fc&1
                                tmp_bin_fc = tmp_bin_fc - LSB_fc
                                rram_map_fc[j][k] = tmp_bin_fc
                            w_BD_fc[j] = ((rram_map_fc[j]-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_BD_fc
                    else:
                        heat_value = (int(sector[s][1])/16)*(2**SynapseBit)
                        print('origitnal heat = ', sector[s][1])
                        print('heat value = ', heat_value)
                        tmp_w3 = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        rram_map = torch.zeros(int(tmp_w3.shape[0]),int(tmp_w3.shape[1]))
                        for j in range(tmp_w3.shape[0]):
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w3[j]-RealMax) + NormalizedMax).round()
                            D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                            resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                            tmp_w3[j]=resumedata
                        w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w3
            param.data = w.reshape(param.data.shape)







def placement(model,args):


    print('================  place from corner ==================')
    x=1
    y=0
    z=1
    sector = []
    tmp = [(0,0)]
    sector.append(tmp)
    while z<=53:
        if (x==z)and(y<z) :
            tmp = [(x,y)]
            sector.append(tmp)
            y=y+1
        if (x==z)and(y==z) :
            tmp = [(x,y)]
            sector.append(tmp)
            x=x-1
        if (x<z)and(y==z) :
            if (x!=0) and (y==z):
                tmp = [(x,y)]
                sector.append(tmp)
                x=x-1
            if (x==0) and (y==z):
                tmp = [(x,y)]
                sector.append(tmp)
                z = z+1
                x=z
                y=0



    print('================== start placement in shan shape =================')


    print('======================== place subarrays of all layer===========================')

    ######### heat map, subarray placed  start from top left ############
    if args.direct=='tl':
        #f = open('dr_heatmap.txt','r')
        f = open('heatmap120W.txt','r')
        #f = open('testheatmap120W.txt','r')
        words = f.read()
        f.close()
        k_limit = 54
        heatmap=[]
        for k in range(len(words)):
            if k == k_limit :
                k_limit = k_limit + 55
            else :
                heatmap.append(words[k])

        print('map weight to subarray from top left')
        for i in range(len(sector)):
            idx = sector[i][0]
            heat_idx = 54*idx[1] + idx[0]
            sector[i].append(heatmap[heat_idx])
        print(sector)
    #######################################################################

    ######## heat map , subarray placed start from top right#############
    if args.direct == 'tr':
        #f = open('dr_heatmap.txt', 'r')
        f = open('heatmap120W.txt', 'r')
        #f = open('testheatmap120W.txt','r')
        heatmap=[]
        for line in f.readlines():
            reverse_line = line[::-1]
            for i in range(len(line)):
                if i != 0 :
                    heatmap.append(reverse_line[i])
        f.close()
        print('map weight to subarray from top right')
        print('len heatmap = ', len(heatmap))
        for i in range(len(sector)):
            idx = sector[i][0]
            heat_idx = 54*idx[1] + idx[0]
            sector[i].append(heatmap[heat_idx])
        print(sector)
    ######################################################################

    ######### heat map , subarray placed start from bottom right#############
    if args.direct == 'br':
        #f = open('dr_heatmap.txt', 'r')
        f = open('heatmap120W.txt', 'r')
        #f = open('testheatmap120W.txt','r')
        words = f.read()
        f.close()
        rev = words[::-1]
        heatmap=[]
        k_limit=0
        for k in range(len(rev)):
            if k==k_limit:
                k_limit=k_limit+55
            else:
                heatmap.append(rev[k])
        print('map weight to subarray from bottom right')
        print('len of heatmap = ', len(heatmap))

        for i in range(len(sector)):
            idx = sector[i][0]
            heat_idx = 54*idx[1] + idx[0]
            sector[i].append(heatmap[heat_idx])
    ######################################################################

    ######### heat map , subarray placed start from bottom left#############
    if args.direct == 'bl':
        f = open('heatmap120W.txt', 'r')
        ##f = open('dr_heatmap.txt', 'r')
        #f = open('testheatmap120W.txt','r')
        words = f.read()
        f.close()
        rev = words[::-1]
        f = open('bottom_left_heatmap.txt','w')
        #f = open('bottom_left_zeroheatmap120W.txt','w')
        f.write(rev)
        f.close()
        f = open('bottom_left_heatmap.txt','r')
        heatmap=[]
        idx=0
        for line in f.readlines():
            if idx!=0:
                reverse_line = line[::-1]
                for i in range(len(reverse_line)):
                    if idx!=54:
                        if i != 0 :
                            heatmap.append(reverse_line[i])
                    else:
                        heatmap.append(reverse_line[i])
            idx=idx+1
        f.close()
        print('map weight to subarray from bottom left')
        print('len of heatmap = ', len(heatmap))

        for i in range(len(sector)):
            idx = sector[i][0]
            heat_idx = 54*idx[1] + idx[0]
            sector[i].append(heatmap[heat_idx])
    #######################################################################


    realmin=1000
    realmax=-9999
    for name, param in (model.named_parameters()):
        if len(param.data.size()) != 1:
            w = param.data

            if torch.min(param.data)<realmin:
                realmin = torch.min(param.data)
            if torch.max(param.data)>realmax:
                realmax = torch.max(param.data)

    print('realmin = ', realmin)
    print('realmax = ', realmax)



    SynapseBit = 8
    CellBit = 4
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = realmin
    RealMax = realmax
    SubarrayRow = 128
    SubarrayCol = 128
    print('test')

    #level = 3

    subarray_list = []
    all_layer_len = []
    each_layer_len = []
    for name, param in (model.named_parameters()):

        if len(param.data.size()) == 4 :
            row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            col = param.data.shape[0]*numColPerSynapse
            w = param.data.reshape(param.data.shape[0],row)

            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))

                    G_map = torch.zeros(int(rram_partition[i,j].shape[0]*numColPerSynapse),int(rram_partition[i,j].shape[1]))
                    for k in range(rram_partition[i,j].shape[0]):
                        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(rram_partition[i,j][k]-RealMax) + NormalizedMax).round()
                        value = newdata
                        cellrange = 2**CellBit
                        for m in range(numColPerSynapse):
                            reminder = torch.ceil(value%cellrange)
                            value = torch.floor(value/cellrange)
                            G_map[(numColPerSynapse)*k+m] =reminder

                    #G_initial = torch.ones(G_map.shape) * 3.07 ## conductance range 3.07nS ~ 38.4nS
                    #G_level = G_map * 2.3553 ## each level is 2.3553nS = (38.4-3.07)/15
                    #G_value = G_initial + G_level

                    G_initial = torch.ones(G_map.shape) * 100 ## conductance range 100nS ~ 5000nS
                    G_level = G_map * 326.66 ## each level is 16333.3 = (5000-100)/15
                    G_value = G_initial + G_level

                    if param.data.shape[0]>=128 :
                        G_value1 = torch.sum(G_value[0:128 ,0:128])
                        G_value2 = torch.sum(G_value[128:256, 0:128])
                    else :
                        G_value1 = torch.sum(G_value[0:param.data.shape[0], 0:128])
                        G_value2 = torch.sum(G_value[param.data.shape[0]:param.data.shape[0]*numColPerSynapse, 0:128])


                    key = (i,j)
                    value = rram_partition_total_sum[i,j]
                    tmp = [key, value, name, G_value1, G_value2 ]
                    rram_partition_total_sum_list.append(tmp)


            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])
            #rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1], reverse = True)


            subarray_list.append(rram_partition_total_sum_list)
            each_layer_len.append(len(rram_partition_total_sum_list))





        if len(param.data.size()) == 2 :
            row = param.data.shape[1]
            col = param.data.shape[0]*numColPerSynapse
            w = param.data.reshape(param.data.shape[0],row)

            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))

                    G_map = torch.zeros(int(rram_partition[i,j].shape[0]*numColPerSynapse),int(rram_partition[i,j].shape[1]))
                    for k in range(rram_partition[i,j].shape[0]):
                        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(rram_partition[i,j][k]-RealMax) + NormalizedMax).round()
                        value = newdata
                        cellrange = 2**CellBit
                        for m in range(numColPerSynapse):
                            reminder = torch.ceil(value%cellrange)
                            value = torch.floor(value/cellrange)
                            G_map[(numColPerSynapse)*k+m] =reminder

                    #G_initial = torch.ones(G_map.shape) * 3.07 ## conductance range 3.07nS ~ 38.4nS
                    #G_level = G_map * 2.3553 ## each level is 2.3553nS = (38.4-3.07)/15
                    #G_value = G_initial + G_level

                    G_initial = torch.ones(G_map.shape) * 100 ## conductance range 100nS ~ 5000nS
                    G_level = G_map * 326.66 ## each level is 326.66 = (5000-100)/15
                    G_value = G_initial + G_level


                    if param.data.shape[0]>=128 :
                        G_value1 = torch.sum(G_value[0:128 ,0:128])
                        G_value2 = torch.sum(G_value[128:256, 0:128])
                    else :
                        G_value1 = torch.sum(G_value[0:param.data.shape[0], 0:128])
                        G_value2 = torch.sum(G_value[param.data.shape[0]:param.data.shape[0]*numColPerSynapse, 0:128])

                    key = (i,j)
                    value = rram_partition_total_sum[i,j]
                    tmp = [key, value, name, G_value1, G_value2 ]
                    rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])
            #rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1],reverse=True)

            subarray_list.append(rram_partition_total_sum_list)

            each_layer_len.append(len(rram_partition_total_sum_list))



    ##### sector_format [sector_key, sector_heat_value, subarray_key, subarray_value, subarray_layer, first_4bit_sub_Glevel, second_4bit_sub_Glevel, first or second 4bit sub]

    if args.tile_pairing==0:
    ############ place subarray in sector with my method ################################
        redundant_heatlist=[]
        if args.experiment == 0:
            tmplist=[]
            for i in range(len(subarray_list)):
                for j in range(len(subarray_list[i])):
                    sub_key = subarray_list[i][j][0]
                    sub_value = subarray_list[i][j][1]
                    layer_name = subarray_list[i][j][2]
                    G_level_sub1 = subarray_list[i][j][3]
                    G_level_sub2 = subarray_list[i][j][4]
                    tmp1 = [sub_key, sub_value, layer_name, G_level_sub1, 1]
                    tmp2 = [sub_key, sub_value, layer_name, G_level_sub2, 2]
                    tmplist.append(tmp1)
                    tmplist.append(tmp2)

            for i in range(len(tmplist)):
                sector[i].append(tmplist[i][0])
                sector[i].append(tmplist[i][1])
                sector[i].append(tmplist[i][2])
                sector[i].append(tmplist[i][3])
                sector[i].append(tmplist[i][4])
        else:
            tmplist=[]
            idx=0
            for i in range(len(subarray_list)):
                for j in range(len(subarray_list[i])):
                    sub_key = subarray_list[i][j][0]
                    sub_value = subarray_list[i][j][1]
                    layer_name = subarray_list[i][j][2]
                    G_level_sub1 = subarray_list[i][j][3]
                    tmp = [sub_key, sub_value, layer_name, G_level_sub1, 1]
                    tmplist.append(tmp)
            for i in range(len(tmplist)):
                sector[i].append(tmplist[i][0])
                sector[i].append(tmplist[i][1])
                sector[i].append(tmplist[i][2])
                sector[i].append(tmplist[i][3])
                sector[i].append(tmplist[i][4])
                idx=idx+1

    else:
    ############ place subarray in sector using tile pairing ##########################
        tmplist=[]
        for i in range(len(subarray_list)):
            for j in range(len(subarray_list[i])):
                sub_key = subarray_list[i][j][0]
                sub_value = subarray_list[i][j][1]
                layer_name = subarray_list[i][j][2]
                tmp=[sub_key, sub_value, layer_name]
                tmplist.append(tmp)
        idx=0
        tmp_idx =0
        print('each_layer_len = ', each_layer_len)
        redundant_heatlist = []
        while tmp_idx <  len(tmplist):
            for i in range(len(each_layer_len)):
                redundant_tile = (math.ceil(each_layer_len[i]*0.1))
                print('redundant_tile=',redundant_tile)
                for d in range(redundant_tile):
                    sector[idx].append((99,99))
                    sector[idx].append(16)
                    sector[idx].append('redundant')
                    sector[idx].append(tmplist[tmp_idx][2])
                    t = (idx,sector[idx][1], tmplist[tmp_idx][2])
                    redundant_heatlist.append(t)
                    print('redundant_idx = ', idx)
                    idx=idx+1
                for j in range((each_layer_len[i])):
                    sector[idx].append(tmplist[tmp_idx][0])
                    sector[idx].append(tmplist[tmp_idx][1])
                    sector[idx].append(tmplist[tmp_idx][2])
                    tmp_idx = tmp_idx + 1
                    idx = idx + 1
        print('redundant_heatlist = ', redundant_heatlist)
        print('idx=',idx)
        print(sector)
    ###################################################################################


    #total_heat = 0
    #for i in range(len(sector)):
    #    if sector[i][2][2] == 'features.module.0.weight':
    #        total_heat = total_heat + int(sector[i][1])
    #    if sector[i][2][2] == 'features.module.2.weight':
    #        total_heat = total_heat + int(sector[i][1])
    #    if sector[i][2][2] == 'features.module.5.weight':
    #        total_heat = total_heat + int(sector[i][1])
    #    if sector[i][2][2] == 'features.module.7.weight':
    #        total_heat = total_heat + int(sector[i][1])
    #    if sector[i][2][2] == 'features.module.10.weight':
    #        total_heat = total_heat + int(sector[i][1])
    #    if sector[i][2][2] == 'features.module.12.weight':
    #        total_heat = total_heat + int(sector[i][1])

    print(sector)
    #
    #print('conv layer total_heat in this floorplan = ', total_heat)


    return sector, len(tmplist), redundant_heatlist, idx



def weight_sensitivity(_model, args):
    realmin = 1000
    realmax = -9999
    for name, param in (_model.named_parameters()):
        if len(param.data.size()) != 1:
            w = param.data
            if torch.min(param.data) < realmin:
                realmin = torch.min(param.data)
            if torch.max(param.data) > realmax:
                realmax = torch.max(param.data)

    print('realmin = ', realmin)
    print('realmax = ', realmax)
    SynapseBit = args.testbit
    CellBit = args.testbit
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = realmin
    RealMax = realmax
    SubarrayRow = 128
    SubarrayCol = 128

    for name, param in (_model.named_parameters()):
        if len(param.data.size()) == 4 :
            print('name=',name)
            row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])



            if args.arch == 'vgg11':
                ratio= int(args.affect_ratio)/100
                #if name == 'module.features.module.3.weight':
                #    #ratio = 0.6
                #    affected_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                #    print('layer2 have ',len(rram_partition_total_sum),' subarrays')
                #    print('layer2 affected ',affected_subarray,' subarrays. ')
                #    for i in range(affected_subarray):
                #        key,value = rram_partition_total_sum_list[i]
                #        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                #        heat_value = (7/16) * (2**(int(args.testbit)))
                #        for j in range(tmp_w.shape[0]):
                #            newdata = (((NormalizedMax - NormalizedMin) / (RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                #            D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                #            resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                #            tmp_w[j]=resumedata
                #    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                #    print('layer2 pruned percentage = ', affected_subarray/len(rram_partition_total_sum))

                #if name == 'module.features.module.6.weight':
                #    #ratio = 0.6
                #    affected_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                #    print('layer3 have ',len(rram_partition_total_sum),' subarrays')
                #    print('layer3 affected ',affected_subarray,' subarrays. ')
                #    #for i in range(affected_subarray):
                #    for i in range(1):
                #        key,value = rram_partition_total_sum_list[i]
                #        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                #        heat_value = (7/16) * (2**(int(args.testbit)))
                #        for j in range(tmp_w.shape[0]):
                #            newdata = (((NormalizedMax - NormalizedMin) / (RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                #            D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                #            resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                #            tmp_w[j]=resumedata
                #    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                #    print('layer3 pruned percentage = ', affected_subarray/len(rram_partition_total_sum))

                #if name == 'module.features.module.8.weight':
                #    #ratio = 0.6
                #    affected_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                #    print('layer4 have ',len(rram_partition_total_sum),' subarrays')
                #    print('layer4 affected ',affected_subarray,' subarrays. ')
                #    for i in range(affected_subarray):
                #        key,value = rram_partition_total_sum_list[i]
                #        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                #        heat_value = (7/16) * (2**(int(args.testbit)))
                #        for j in range(tmp_w.shape[0]):
                #            newdata = (((NormalizedMax - NormalizedMin) / (RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                #            D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                #            resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                #            tmp_w[j]=resumedata
                #    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                #    print('layer4 pruned percentage = ', affected_subarray/len(rram_partition_total_sum))


                #if name == 'module.features.module.11.weight':
                #    #ratio = 0.6
                #    affected_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                #    print('layer5 have ',len(rram_partition_total_sum),' subarrays')
                #    print('layer5 affected ',affected_subarray,' subarrays. ')
                #    for i in range(affected_subarray):
                #        key,value = rram_partition_total_sum_list[i]
                #        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                #        heat_value = (7/16) * (2**(int(args.testbit)))
                #        for j in range(tmp_w.shape[0]):
                #            newdata = (((NormalizedMax - NormalizedMin) / (RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                #            D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                #            resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                #            tmp_w[j]=resumedata
                #    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                #    print('layer5 pruned percentage = ', affected_subarray/len(rram_partition_total_sum))

                #if name == 'module.features.module.13.weight':
                #    #ratio = 0.6
                #    affected_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                #    print('layer6 have ',len(rram_partition_total_sum),' subarrays')
                #    print('layer6 affected ',affected_subarray,' subarrays. ')
                #    for i in range(affected_subarray):
                #        key,value = rram_partition_total_sum_list[i]
                #        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                #        heat_value = (7/16) * (2**(int(args.testbit)))
                #        for j in range(tmp_w.shape[0]):
                #            newdata = (((NormalizedMax - NormalizedMin) / (RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                #            D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                #            resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                #            tmp_w[j]=resumedata
                #    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                #    print('layer6 pruned percentage = ', affected_subarray/len(rram_partition_total_sum))


                #if name == 'module.features.module.16.weight':
                #    #ratio = 0.6
                #    affected_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                #    print('layer7 have ',len(rram_partition_total_sum),' subarrays')
                #    print('layer7 affected ',affected_subarray,' subarrays. ')
                #    for i in range(affected_subarray):
                #        key,value = rram_partition_total_sum_list[i]
                #        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                #        heat_value = (6/16) * (2**(int(args.testbit)))
                #        for j in range(tmp_w.shape[0]):
                #            newdata = (((NormalizedMax - NormalizedMin) / (RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                #            D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                #            resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                #            tmp_w[j]=resumedata
                #    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                #    print('layer7 pruned percentage = ', affected_subarray/len(rram_partition_total_sum))

                #if name == 'module.features.module.18.weight':
                #    #ratio = 0.6
                #    affected_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                #    print('layer8 have ',len(rram_partition_total_sum),' subarrays')
                #    print('layer8 affected ',affected_subarray,' subarrays. ')
                #    for i in range(affected_subarray):
                #        key,value = rram_partition_total_sum_list[i]
                #        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                #        heat_value = (6/16) * (2**(int(args.testbit)))
                #        for j in range(tmp_w.shape[0]):
                #            newdata = (((NormalizedMax - NormalizedMin) / (RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                #            D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                #            resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                #            tmp_w[j]=resumedata
                #    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                #    print('layer8 pruned percentage = ', affected_subarray/len(rram_partition_total_sum))


            param.data = w.reshape(param.data.shape)


        if len(param.data.size()) == 2 :
            print('name=',name)
            row = param.data.shape[1]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])


            if args.arch == 'vgg11':
                ratio= int(args.affect_ratio)/100
                #if name == 'module.classifier.0.weight':
                #    #ratio = 0.6
                #    affected_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                #    print('layer9 have ',len(rram_partition_total_sum),' subarrays')
                #    print('layer9 affected ',affected_subarray,' subarrays. ')
                #    for i in range(affected_subarray):
                #        key,value = rram_partition_total_sum_list[i]
                #        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                #        heat_value = (6/16) * (2**(int(args.testbit)))
                #        for j in range(tmp_w.shape[0]):
                #            newdata = (((NormalizedMax - NormalizedMin) / (RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                #            D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                #            resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                #            tmp_w[j]=resumedata
                #    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                #    print('layer9 pruned percentage = ', affected_subarray/len(rram_partition_total_sum))

                #if name == 'module.classifier.2.weight':
                #    #ratio = 0.6
                #    affected_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                #    print('layer10 have ',len(rram_partition_total_sum),' subarrays')
                #    print('layer10 affected ',affected_subarray,' subarrays. ')
                #    for i in range(affected_subarray):
                #        key,value = rram_partition_total_sum_list[i]
                #        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                #        heat_value = (6/16) * (2**(int(args.testbit)))
                #        for j in range(tmp_w.shape[0]):
                #            newdata = (((NormalizedMax - NormalizedMin) / (RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                #            D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                #            resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                #            tmp_w[j]=resumedata
                #    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                #    print('layer10 pruned percentage = ', affected_subarray/len(rram_partition_total_sum))


                if name == 'module.classifier.4.weight':
                    #ratio = 0.6
                    affected_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                    print('layer11 have ',len(rram_partition_total_sum),' subarrays')
                    print('layer11 affected ',affected_subarray,' subarrays. ')
                    for i in range(affected_subarray):
                        key,value = rram_partition_total_sum_list[i]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        heat_value = (6/16) * (2**(int(args.testbit)))
                        for j in range(tmp_w.shape[0]):
                            newdata = (((NormalizedMax - NormalizedMin) / (RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                            resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                            tmp_w[j]=resumedata
                    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                    print('layer11 pruned percentage = ', affected_subarray/len(rram_partition_total_sum))




def model_acc_under_thermal_impact(_model, args):
    realmin=1000
    realmax=-9999
    for name, param in (_model.named_parameters()):
        if len(param.data.size()) != 1:
            w = param.data
            if torch.min(param.data)<realmin:
                realmin = torch.min(param.data)
            if torch.max(param.data)>realmax:
                realmax = torch.max(param.data)
    print('realmin = ', realmin)
    print('realmax = ', realmax)
    SynapseBit = int(args.testbit)
    CellBit = SynapseBit
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = realmin
    RealMax = realmax
    SubarrayRow = 128
    SubarrayCol = 128


    heat_value = 0
    if args.temperature == 300 :
        heat_value = 0
    if args.temperature == 310 :
        heat_value = 1
    if args.temperature == 320 :
        heat_value = 2
    if args.temperature == 330:
        heat_value = 3
    if args.temperature == 340:
        heat_value = 4
    if args.temperature == 350:
        heat_value = 8
    if args.temperature == 360:
        heat_value = 10
    if args.temperature == 370:
        heat_value = 12
    if args.temperature == 380:
        heat_value = 14
    if args.temperature == 390:
        heat_value = 15
    if args.temperature == 400:
        heat_value = 16


    print('in new function~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    for name, param in (_model.named_parameters()):
        if len(param.data.size()) == 4 :
            print('name=',name)
            row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    #heat = heat_value/16 * 2**CellBit
                    heat = heat_value/32 * 2**CellBit
                    #heat = heat_value
                    print('heat_value = ', heat_value)
                    print('heat = ', heat)
                    for k in range(rram_partition[i,j].shape[0]):
                        #_max = realmax
                        _max = torch.max(rram_partition[i,j][k])
                        #_min = realmin
                        _min = torch.min(rram_partition[i,j][k])
                        h_impact = (_max-_min)*(((2**CellBit)-1-heat)/(2**CellBit-1))
                        new_tmp = (rram_partition[i,j][k]>(h_impact+_min)).float()*(h_impact+_min) + (rram_partition[i,j][k]<=(h_impact+_min)).float()*rram_partition[i,j][k]
                        rram_partition[i,j][k]=new_tmp

                    w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow] = rram_partition[i,j]
            param.data = w.reshape(param.data.shape)


        if len(param.data.size()) == 2 :
            print('name=',name)
            row = param.data.shape[1]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    #heat = heat_value/16 * 2**CellBit
                    heat = heat_value/32 * 2**CellBit
                    #heat = heat_value

                    for k in range(rram_partition[i,j].shape[0]):
                        #_max = realmax
                        _max = torch.max(rram_partition[i,j][k])
                        #_min = realmin
                        _min = torch.min(rram_partition[i,j][k])
                        h_impact = (_max-_min)*(((2**CellBit)-1-heat)/(2**CellBit-1))
                        new_tmp = (rram_partition[i,j][k]>(h_impact+_min)).float()*(h_impact+_min) + (rram_partition[i,j][k]<=(h_impact+_min)).float()*rram_partition[i,j][k]
                        rram_partition[i,j][k]=new_tmp


                    w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow] = rram_partition[i,j]
            param.data = w.reshape(param.data.shape)




def remapping_thermal(_model, sector, args, len_tmp_list, redundant_heatlist, idx): ##???????????????tile????????????????????????????????????????????????
    realmin=1000
    realmax=-9999
    for name, param in (_model.named_parameters()):
        if len(param.data.size()) != 1:
            w = param.data
            if torch.min(param.data)<realmin:
                realmin = torch.min(param.data)
            if torch.max(param.data)>realmax:
                realmax = torch.max(param.data)
    print('realmin = ', realmin)
    print('realmax = ', realmax)
    SynapseBit = int(args.testbit)
    CellBit = SynapseBit
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = realmin
    RealMax = realmax
    SubarrayRow = 128
    SubarrayCol = 128


    print('idx=', idx)

    for name, param in (_model.named_parameters()):
        if len(param.data.size()) == 4 :
            print('name=',name)
            row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])


            for i in range(idx):
                if sector[i][4] == name:
                    key = sector[i][2]
                    print('sextor = ', sector[i])
                    print('key=', key)
                    #print('key[0]=',key[0], 'key[1]=',key[1])
                    tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                    heat_value = int(sector[i][1])/16 * 2**CellBit
                    print('heat=', sector[i][1])
                    print('heat real =', heat_value)

                    for j in range(tmp_w.shape[0]):
                        _max = torch.max(tmp_w[j])
                        _min = torch.min(tmp_w[j])
                        h_impact = (_max-_min)*(((2**CellBit)-1-heat_value)/(2**CellBit-1))
                        new_tmp_pos = (tmp_w[j]>(h_impact+_min)).float()*(h_impact+_min) + (tmp_w[j]<=(h_impact+_min)).float()*tmp_w[j]
                        tmp_w[j]=new_tmp_pos
                    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
            param.data = w.reshape(param.data.shape)


        if len(param.data.size()) == 2 :
            print('name=',name)
            row = param.data.shape[1]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])

            for i in range(idx):
                if sector[i][4] == name:
                    key = sector[i][2]
                    tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                    heat_value = int(sector[i][1])/16 * 2**CellBit
                    for j in range(tmp_w.shape[0]):
                        _max = torch.max(tmp_w[j])
                        _min = torch.min(tmp_w[j])
                        h_impact = (_max-_min)*(((2**CellBit)-1-heat_value)/(2**CellBit-1))
                        new_tmp_pos = (tmp_w[j]>(h_impact+_min)).float()*(h_impact+_min) + (tmp_w[j]<=(h_impact+_min)).float()*tmp_w[j]
                        tmp_w[j]=new_tmp_pos




                    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
            param.data = w.reshape(param.data.shape)





def tile_pairing(_model, sector, args, len_tmp_list, redundant_heatlist, idx): ##???????????????tile????????????????????????????????????????????????
    realmin=1000
    realmax=-9999
    for name, param in (_model.named_parameters()):
        if len(param.data.size()) != 1:
            w = param.data
            if torch.min(param.data)<realmin:
                realmin = torch.min(param.data)
            if torch.max(param.data)>realmax:
                realmax = torch.max(param.data)
    print('realmin = ', realmin)
    print('realmax = ', realmax)
    SynapseBit = args.testbit
    CellBit = args.testbit
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = realmin
    RealMax = realmax
    SubarrayRow = 128
    SubarrayCol = 128

    for name, param in (_model.named_parameters()):
        if len(param.data.size()) == 4 :
            print('name=',name)
            row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])


            ## ????????????5bit weight???5bit cellresolution??????????????????????????? 5????????????
            ## ?????????weight map???subarray??????????????????????????????redundant tile?????????
            for i in range(idx):
                if sector[i][4] == name:
                    key = sector[i][2]
                    print('sextor = ', sector[i])
                    print('key=', key)
                    print('key[0]=',key[0], 'key[1]=',key[1])
                    tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                    heat_value = int(sector[i][1])/16 * 2**CellBit
                    print('heat=', sector[i][1])
                    print('heat real =', heat_value)

                    for j in range(tmp_w.shape[0]):
                        newdata = (((NormalizedMax - NormalizedMin) / (RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                        D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                        resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                        tmp_w[j]=resumedata
                    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
            param.data = w.reshape(param.data.shape)


            #paired_key= []
            #for i in range(len(redundant_heatlist)):
            #    if redundant_heatlist[i][2] == name:
            #        paired_sub = choice(rram_partition_total_sum_list)
            #        print('paird_sub = ', paired_sub)
            #        key = paired_sub[0]
            #        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
            #        paired_key.append(key)
            #        print('paired_key=',paired_key)
            #        for j in range(len(sector)):
            #            if j < idx:
            #                if sector[j][4]==name and sector[j][2]==key:
            #                    heat_value = int(sector[j][1])/16*(2**CellBit)
            #                    print('redundant_heat_list_i=', redundant_heatlist[i])
            #                    heat_value_pair = int(redundant_heatlist[i][1])/16*(2**CellBit)
            #                    #heat_value_pair = 6/16*(2**CellBit)
            #                    for k in range(0,tmp_w.shape[0],2):
            #                        newdata1 = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[k]-RealMax) + NormalizedMax).round()
            #                        newdata2 = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[k+1]-RealMax) + NormalizedMax).round()
            #                        D1 = ((newdata1)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata1)<=((2**CellBit)-heat_value -1)).float()*newdata1
            #                        D2 = ((newdata2)>((2**CellBit)-heat_value_pair-1)).float()*((2**CellBit)-heat_value_pair-1) + ((newdata2)<=(2**CellBit)-heat_value_pair-1).float()*newdata2
            #                        resume1 = ((D1-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
            #                        resume2 = ((D2-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
            #                        tmp_w[k]=resume1
            #                        tmp_w[k+1]=resume2

            #                    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
            #skip=0
            #for i in range(idx):
            #    print('idx = ', i)
            #    skip=0
            #    if sector[i][4] == name:
            #        key = sector[i][2]
            #        print('key=',key)
            #        print('paired_key = ', paired_key)
            #        for x in range(len(paired_key)):
            #            if key == paired_key[x]:
            #                skip = 1
            #
            #        if skip==0:
            #            if int(sector[i][1])==5 or int(sector[i][1])==6 or int(sector[i][1]) == 4 :
            #                print('bitwidth downgrading')
            #                tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
            #                rram_map = torch.zeros(int(tmp_w.shape[0]),int(tmp_w.shape[1]))
            #                for j in range(tmp_w.shape[0]):
            #                    newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
            #                    rram_map[j] = newdata
            #                    w_BD = torch.zeros(tmp_w.shape)
            #                for j in range(rram_map.shape[0]):
            #                    for k in range(len(rram_map[j])):
            #                        tmp_bin = int(rram_map[j][k])
            #                        LSB = tmp_bin&1
            #                        tmp_bin = tmp_bin - LSB
            #                        rram_map[j][k] = tmp_bin
            #                    w_BD[j] = ((rram_map[j]-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
            #                w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_BD
            #            else:
            #                print('normal thermal')
            #                heat_value = (int(sector[i][1])/16)*(2**CellBit)
            #                tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
            #                for j in range(tmp_w.shape[0]):
            #                    newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
            #                    D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
            #                    resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
            #                    tmp_w[j]=resumedata
            #                w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
            #param.data = w.reshape(param.data.shape)


        if len(param.data.size()) == 2 :
            print('name=',name)
            row = param.data.shape[1]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])

            for i in range(idx):
                if sector[i][4] == name:
                    key = sector[i][2]
                    tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                    heat_value = int(sector[i][1])/16 * 2**CellBit
                    for j in range(tmp_w.shape[0]):
                        newdata = (((NormalizedMax - NormalizedMin) / (RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                        D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
                        resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
                        tmp_w[j]=resumedata
                    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
            param.data = w.reshape(param.data.shape)


            #paired_key= []
            #for i in range(len(redundant_heatlist)):
            #    if redundant_heatlist[i][2] == name:
            #        paired_sub = choice(rram_partition_total_sum_list)
            #        print('paird_sub = ', paired_sub)
            #        key = paired_sub[0]
            #        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
            #        paired_key.append(key)
            #        print('paired_key=',paired_key)
            #        for j in range(len(sector)):
            #            if j < idx:
            #                if sector[j][4]==name and sector[j][2]==key:
            #                    heat_value = int(sector[j][1])/16*(2**CellBit)
            #                    print('redundant_heat_list_i=', redundant_heatlist[i])
            #                    heat_value_pair = int(redundant_heatlist[i][1])/16*(2**CellBit)
            #                    #heat_value_pair = 6/16*(2**CellBit)
            #                    for k in range(0,tmp_w.shape[0],2):
            #                        newdata1 = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[k]-RealMax) + NormalizedMax).round()
            #                        newdata2 = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[k+1]-RealMax) + NormalizedMax).round()
            #                        D1 = ((newdata1)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata1)<=((2**CellBit)-heat_value -1)).float()*newdata1
            #                        D2 = ((newdata2)>((2**CellBit)-heat_value_pair-1)).float()*((2**CellBit)-heat_value_pair-1) + ((newdata2)<=(2**CellBit)-heat_value_pair-1).float()*newdata2
            #                        resume1 = ((D1-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
            #                        resume2 = ((D2-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
            #                        tmp_w[k]=resume1
            #                        tmp_w[k+1]=resume2

            #                    w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w


            #skip=0
            #for i in range(idx):
            #    print('idx = ', i)
            #    skip=0
            #    if sector[i][4] == name:
            #        key = sector[i][2]
            #        print('key=',key)
            #        print('paired_key = ', paired_key)
            #        for x in range(len(paired_key)):
            #            if key == paired_key[x]:
            #                skip = 1

            #        if skip==0:
            #            if int(sector[i][1])==5 or int(sector[i][1])==6 or int(sector[i][1]) == 4  :
            #                print('bitwidth downgrading')
            #                tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
            #                rram_map = torch.zeros(int(tmp_w.shape[0]),int(tmp_w.shape[1]))
            #                for j in range(tmp_w.shape[0]):
            #                    newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
            #                    rram_map[j] = newdata
            #                    w_BD = torch.zeros(tmp_w.shape)
            #                for j in range(rram_map.shape[0]):
            #                    for k in range(len(rram_map[j])):
            #                        tmp_bin = int(rram_map[j][k])
            #                        LSB = tmp_bin&1
            #                        tmp_bin = tmp_bin - LSB
            #                        rram_map[j][k] = tmp_bin
            #                    w_BD[j] = ((rram_map[j]-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
            #                w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_BD
            #            else:
            #                print('normal thermal')
            #                heat_value = (int(sector[i][1])/16)*(2**CellBit)
            #                tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
            #                for j in range(tmp_w.shape[0]):
            #                    newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
            #                    D = ((newdata)>((2**CellBit)-heat_value-1)).float()*((2**CellBit)-heat_value-1) + ((newdata)<=((2**CellBit)-heat_value -1)).float()*newdata
            #                    resumedata = ((D-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin) + RealMax
            #                    tmp_w[j]=resumedata
            #                w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
            #param.data = w.reshape(param.data.shape)





def split_after_prune(_model, sector, args):
    print('---------------- start splitting and map into subarrays--------------------------')
    realmin=1000
    realmax=-9999
    for name, param in (_model.named_parameters()):
        if len(param.data.size()) != 1:
            w = param.data
            if torch.min(param.data)<realmin:
                realmin = torch.min(param.data)
            if torch.max(param.data)>realmax:
                realmax = torch.max(param.data)
    print('realmin = ', realmin)
    print('realmax = ', realmax)
    SynapseBit = 8
    CellBit = 4
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = realmin
    RealMax = realmax
    SubarrayRow = 128
    SubarrayCol = 128
    cell_limit = 7

    place_idx=0

    for name, param in (_model.named_parameters()):
        if len(param.data.size()) == 4 :
            print('name=',name)
            row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_row_sum = {} ## a row sum in subarray
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1], reverse = True)

            #print('sector len = ', len(sector))
            #print('len of rram parititon totalt sum list = ', len(rram_partition_total_sum_list))
            ratio = int(args.prune_ratio)/100
            print('ratio=',ratio)
            #split_num = math.floor(len(rram_partition_total_sum_list) * ratio)
            #t_len = math.floor(len(rram_partition_total_sum_list)* (1-ratio))

            split_num=0
            t_len=0
            ratio_bigger_than50 = 0
            if ratio<=0.5:
                split_num = math.floor(len(rram_partition_total_sum_list)*ratio)
                #split_num = math.floor(len(rram_partition_total_sum_list)*(ratio+0.1))
                t_len=math.floor(len(rram_partition_total_sum_list)* (1-ratio))
                #t_len=math.floor(len(rram_partition_total_sum_list)* (1-ratio-0.1))
                ratio_bigger_than50 = 0
                print('split_num=',split_num)
                print('rram_partition_total sum list long = ', len(rram_partition_total_sum_list))
                print('tlen = ', t_len)
            else:
                split_num = math.floor(len(rram_partition_total_sum_list)*(1-ratio))
                t_len=0
                ratio_bigger_than50 = 1
                print('split_num=',split_num)
                print('rram_partition_total sum list long = ', len(rram_partition_total_sum_list))
                print('tlen = ', t_len)



            for i in range(split_num):
                print('i=',i)
                key = rram_partition_total_sum_list[i][0]
                tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                if args.experiment==0:
                    rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse)*2, int(tmp_w.shape[1]))
                    for j in range(tmp_w.shape[0]):
                        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                        value = newdata
                        cellrange = 2**CellBit
                        for k in range(0,2*numColPerSynapse,2):
                            reminder = torch.ceil(value%cellrange)
                            value = torch.floor(value/cellrange)
                            rram_map1_big = (reminder>cell_limit).float()*cell_limit
                            rram_map1_sml = (reminder<=cell_limit).float()*reminder
                            rram_map2_big = (reminder>cell_limit).float()*(reminder-cell_limit)
                            rram_map2_sml = (reminder<=cell_limit).float()*0
                            rram_map[(numColPerSynapse)*(j*2)+k] = rram_map1_big + rram_map1_sml
                            rram_map[(numColPerSynapse)*(j*2)+(k+1)] = rram_map2_big + rram_map2_sml

                    for p in range(1,5,1):
                        print('p=',p)
                        sector[place_idx][2]=key
                        sector[place_idx][3]=torch.sum(torch.abs(rram_map[128*(p-1):128*p, 0:128]))
                        sector[place_idx][4]=name
                        sector[place_idx].append(p/5)##?????????split
                        print('place_idx', place_idx)
                        print('sector[{}]={}'.format(place_idx, sector[place_idx]))
                        place_idx=place_idx+1
                else:
                    cell_limit = (2**int(args.testbit))/2-1
                    rram_map = torch.zeros(int(tmp_w.shape[0])*2 , int(tmp_w.shape[1]))
                    for j in range(tmp_w.shape[0]):
                        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                        rram_map1_big = (newdata > cell_limit).float()*cell_limit
                        rram_map1_sml = (newdata <= cell_limit).float()*newdata
                        rram_map2_big = (newdata > cell_limit).float()*(newdata-cell_limit)
                        rram_map2_sml = (newdata <= cell_limit).float()*0
                        rram_map[j*2] = rram_map1_big + rram_map1_sml
                        rram_map[j*2+1] = rram_map2_big + rram_map2_sml
                    for p in range(1,3,1):
                        print('p=',p)
                        sector[place_idx][2]=key
                        sector[place_idx][3]=torch.sum(torch.abs(rram_map[128*(p-1):128*p, 0:128]))
                        sector[place_idx][4]=name
                        sector[place_idx].append(p/3)##?????????split
                        print('place_idx', place_idx)
                        print('sector[{}]={}'.format(place_idx, sector[place_idx]))
                        place_idx = place_idx + 1



            if ratio_bigger_than50 == 0:
                if args.experiment == 0:
                    for i in range(split_num, t_len ,1):
                        print('i=',i)
                        key = rram_partition_total_sum_list[i][0]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            for k in range(numColPerSynapse):
                                reminder = torch.ceil(value%cellrange)
                                value = torch.floor(value/cellrange)
                                rram_map[(numColPerSynapse)*j+k] = reminder

                        sector[place_idx][2]=key
                        sector[place_idx][3]=torch.sum(torch.abs(rram_map[0:128, 0:128]))
                        sector[place_idx][4]=name
                        sector[place_idx].append(1)
                        print('place_idx=',place_idx)
                        print('sector[{}]={}'.format(place_idx, sector[place_idx]))
                        place_idx=place_idx+1

                        sector[place_idx][2]=key
                        sector[place_idx][3]=torch.sum(torch.abs(rram_map[128:256, 0:128]))
                        sector[place_idx][4]=name
                        sector[place_idx].append(2)
                        print('place_idx=',place_idx)
                        print('sector[{}]={}'.format(place_idx, sector[place_idx]))
                        place_idx=place_idx+1
                else:
                    for i in range(split_num, t_len, 1):
                        print('i=',i)
                        key = rram_partition_total_sum_list[i][0]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        rram_map = torch.zeros(int(tmp_w.shape[0]),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                        sector[place_idx][2]=key
                        sector[place_idx][3]=torch.sum(torch.abs(rram_map[0:128, 0:128]))
                        sector[place_idx][4]=name
                        sector[place_idx].append(1)
                        print('place_idx=',place_idx)
                        print('sector[{}]={}'.format(place_idx, sector[place_idx]))
                        place_idx=place_idx+1



        if len(param.data.size()) == 2 :
            print('name=',name)
            row = param.data.shape[1]
            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_row_sum = {} ## a row sum in subarray
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1], reverse = True)

            #print('sector len = ', len(sector))
            #print('len of rram parititon totalt sum list = ', len(rram_partition_total_sum_list))
            ratio = int(args.prune_ratio)/100
            print('ratio=',ratio)
            #split_num = math.floor(len(rram_partition_total_sum_list) * ratio)
            #t_len = math.floor(len(rram_partition_total_sum_list)* (1-ratio))

            split_num=0
            t_len=0
            ratio_bigger_than50 = 0
            if ratio<=0.5:
                split_num = math.floor(len(rram_partition_total_sum_list)*ratio) ## without redundant tile
                #split_num = math.floor(len(rram_partition_total_sum_list)*(ratio+0.1)) ## with redundant tile
                t_len=math.floor(len(rram_partition_total_sum_list)*(1-ratio))
                #t_len=math.floor(len(rram_partition_total_sum_list)*(1-ratio-0.1))
                ratio_bigger_than50 = 0
                print('split_num=',split_num)
                print('rram_partition_total sum list long = ', len(rram_partition_total_sum_list))
                print('tlen = ', t_len)

            else:
                split_num = math.floor(len(rram_partition_total_sum_list)* (1-ratio))
                #split_num = math.floor(len(rram_partition_total_sum_list)* (1-ratio-0.1))
                t_len=0
                ratio_bigger_than50 = 1
                print('split_num=',split_num)
                print('rram_partition_total sum list long = ', len(rram_partition_total_sum_list))
                print('tlen = ', t_len)



            for i in range(split_num):
                print('i=',i)
                key = rram_partition_total_sum_list[i][0]
                print('sub key = ', key)
                tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                if args.experiment == 0:
                    rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse)*2, int(tmp_w.shape[1]))
                    for j in range(tmp_w.shape[0]):
                        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                        value = newdata
                        cellrange = 2**CellBit
                        for k in range(0,2*numColPerSynapse,2):
                            reminder = torch.ceil(value%cellrange)
                            value = torch.floor(value/cellrange)
                            rram_map1_big = (reminder>cell_limit).float()*cell_limit
                            rram_map1_sml = (reminder<=cell_limit).float()*reminder
                            rram_map2_big = (reminder>cell_limit).float()*(reminder-cell_limit)
                            rram_map2_sml = (reminder<=cell_limit).float()*0
                            rram_map[(numColPerSynapse)*(j*2)+k] = rram_map1_big + rram_map1_sml
                            rram_map[(numColPerSynapse)*(j*2)+(k+1)] = rram_map2_big + rram_map2_sml

                    #print('rram_map,shape=', rram_map.shape)
                    for p in range(1,5,1):
                        print('p=',p)
                        print('place_idx=',place_idx)
                        sector[place_idx][2]=key
                        sector[place_idx][3]=torch.sum(torch.abs(rram_map[128*(p-1):128*p, 0:128]))
                        sector[place_idx][4]=name
                        sector[place_idx].append(p/5)##?????????split
                        print('sector[{}]={}'.format(place_idx, sector[place_idx]))
                        place_idx=place_idx+1
                else:
                    cell_limit = (2**int(args.testbit))/2-1
                    rram_map = torch.zeros(int(tmp_w.shape[0])*2 , int(tmp_w.shape[1]))
                    for j in range(tmp_w.shape[0]):
                        newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                        rram_map1_big = (newdata > cell_limit).float()*cell_limit
                        rram_map1_sml = (newdata <= cell_limit).float()*newdata
                        rram_map2_big = (newdata > cell_limit).float()*(newdata-cell_limit)
                        rram_map2_sml = (newdata <= cell_limit).float()*0
                        rram_map[j*2] = rram_map1_big + rram_map1_sml
                        rram_map[j*2+1] = rram_map2_big + rram_map2_sml
                    for p in range(1,3,1):
                        print('p=',p)
                        sector[place_idx][2]=key
                        sector[place_idx][3]=torch.sum(torch.abs(rram_map[128*(p-1):128*p, 0:128]))
                        sector[place_idx][4]=name
                        sector[place_idx].append(p/3)
                        print('place_idx', place_idx)
                        print('sector[{}]={}'.format(place_idx, sector[place_idx]))
                        place_idx = place_idx + 1


            if ratio_bigger_than50 == 0:
                if args.experiment==0:
                    for i in range(split_num, t_len , 1):
                        print('i=',i)
                        print(name)
                        print(key)
                        key = rram_partition_total_sum_list[i][0]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        rram_map = torch.zeros(int(tmp_w.shape[0]*numColPerSynapse),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                            value = newdata
                            cellrange = 2**CellBit
                            for k in range(numColPerSynapse):
                                reminder = torch.ceil(value%cellrange)
                                value = torch.floor(value/cellrange)
                                rram_map[(numColPerSynapse)*j+k] = reminder
                        sector[place_idx][2]=key
                        sector[place_idx][3]=torch.sum(torch.abs(rram_map[0:128, 0:128]))
                        sector[place_idx][4]=name
                        sector[place_idx].append(1)
                        print('place_idx=',place_idx)
                        print('sector[{}]={}'.format(place_idx, sector[place_idx]))
                        place_idx=place_idx+1

                        sector[place_idx][2]=key
                        sector[place_idx][3]=torch.sum(torch.abs(rram_map[128:256, 0:128]))
                        sector[place_idx][4]=name
                        sector[place_idx].append(2)
                        print('place_idx=',place_idx)
                        print('sector[{}]={}'.format(place_idx, sector[place_idx]))
                        place_idx=place_idx+1
                else:
                    for i in range(split_num, t_len, 1):
                        print('i=',i)
                        key = rram_partition_total_sum_list[i][0]
                        tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                        rram_map = torch.zeros(int(tmp_w.shape[0]),int(tmp_w.shape[1]))
                        for j in range(tmp_w.shape[0]):
                            newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                        sector[place_idx][2]=key
                        sector[place_idx][3]=torch.sum(torch.abs(rram_map[0:128, 0:128]))
                        sector[place_idx][4]=name
                        sector[place_idx].append(1)
                        print('place_idx=',place_idx)
                        print('sector[{}]={}'.format(place_idx, sector[place_idx]))
                        place_idx=place_idx+1


        #print('sector long = ', len(sector))

    print(sector)

    return sector, place_idx




def thermal_after_split(_model, sector, args, len_tmp_list, place_idx):
    print('---------------- thermal effect after splitting--------------------------')
    realmin=1000
    realmax=-9999
    for name, param in (_model.named_parameters()):
        if len(param.data.size()) != 1:
            w = param.data
            if torch.min(param.data)<realmin:
                realmin = torch.min(param.data)
            if torch.max(param.data)>realmax:
                realmax = torch.max(param.data)
    print('realmin = ', realmin)
    print('realmax = ', realmax)
    CellBit = int(args.testbit)
    SynapseBit = CellBit
    numRowPerSynapse = 1
    numColPerSynapse = int(SynapseBit/CellBit)
    NormalizedMin = 0
    NormalizedMax = 2**SynapseBit
    RealMin = realmin
    RealMax = realmax
    SubarrayRow = 128
    SubarrayCol = 128
    cell_limit = 7

    #index=4
    for name, param in (_model.named_parameters()):
        index=0
        if args.experiment==0:
            index=4
        else:
            index=2
        if len(param.data.size()) == 4 or len(param.data.size()) == 2:
            print('name=',name)
            row=0
            if len(param.data.size())==4:
                row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
            if len(param.data.size())==2:
                row = param.data.shape[1]

            w = param.data.reshape(param.data.shape[0],row)
            numRow = math.ceil(row/SubarrayRow)
            numCol = math.ceil(param.data.shape[0]/SubarrayCol)
            rram_partition = {}
            rram_partition_total_sum = {} ## sum of subarray
            rram_partition_total_sum_list = []
            for i in range(numCol):
                for j in range(numRow):
                    rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                    rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))
            for key, value in rram_partition_total_sum.items():
                tmp = [key,value]
                rram_partition_total_sum_list.append(tmp)

            rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1], reverse = True)


            print('rram_partition_total_sum list long is =', len(rram_partition_total_sum_list))
            print('len tmp list = ', len_tmp_list)

            #while index < len(rram_partition_total_sum_list):
            while index < place_idx:
                print('index=',index)
                print('sector{}={}'.format(index,sector[index]))

                if args.experiment == 0:
                    if sector[index][4]==name:
                        _slice = sector[index][7]
                        if _slice ==1:
                            key = sector[index][2]
                            heat_value1 = int(sector[index][1])
                            heat_value2 = int(sector[index+1][1])
                        ####### do thermal effect
                            tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                            print('tmp_w shape = ', tmp_w.shape)
                            rram_map = torch.zeros(int(tmp_w.shape[0])*numColPerSynapse, int(tmp_w.shape[1]))
                            print('rram_map shape =', rram_map.shape)
                            for j in range(tmp_w.shape[0]):
                                #print('j=',j)
                                newdata = (((NormalizedMax-NormalizedMin)/(RealMax-RealMin))*(tmp_w[j]-RealMax) + NormalizedMax).round()
                                value = newdata
                                cellrange = 2**CellBit
                                for k in range(numColPerSynapse):
                                    #print('k=',k)
                                    reminder = torch.ceil(value%cellrange)
                                    value = torch.floor(value/cellrange)
                                    rram_map[(numColPerSynapse)*j+k] = reminder
                            w_prune = torch.zeros(tmp_w.shape)
                            for j in range(0,rram_map.shape[0],(numColPerSynapse)):
                                if numColPerSynapse == 2:
                                    resumedata_prune = (rram_map[j]>(2**CellBit-heat_value1-1)).float() * (2**CellBit-heat_value1-1) + (rram_map[j+1]>(2**CellBit-heat_value2-1)).float() * (2**CellBit-heat_value2-1) * (2**CellBit)
                                    resumedata_noprune = (rram_map[j]<=(2**CellBit-heat_value1-1)).float() * rram_map[j] + (rram_map[j+1]<=(2**CellBit-heat_value2-1)).float() * rram_map[j+1] * (2**CellBit)
                                resumedata = resumedata_prune + resumedata_noprune
                                w_prune[int(j/numColPerSynapse)] = ((resumedata-NormalizedMax)*(RealMax-RealMin))/(NormalizedMax-NormalizedMin)+ RealMax

                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = w_prune
                            index = index + 2
                        else:
                            index = index + 4

                    else:

                        index=index+1
                else:
                    if sector[index][4]==name:
                        _slice = sector[index][7]
                        if _slice == 1:
                            key = sector[index][2]
                            original_hv = int(sector[index][1])
                            heat_value = (int(sector[index][1])/16)*(2**CellBit)
                            print('heat_value=',heat_value)
                            print('origonal hv = ',original_hv)
                            tmp_w = w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                            print('tmp_w shape = ', tmp_w.shape)
                            w_prune = torch.zeros(int(tmp_w.shape[0]), int(tmp_w.shape[1]))
                            for j in range(tmp_w.shape[0]):
                                _max = torch.max(tmp_w[j])
                                _min = torch.min(tmp_w[j])
                                h_impact = (_max-_min)*(((2**CellBit)-1-heat_value)/(2**CellBit-1))
                                new_tmp_pos = (tmp_w[j]>(h_impact+_min)).float()*(h_impact+_min) + (tmp_w[j]<=(h_impact+_min)).float()*tmp_w[j]
                                tmp_w[j]=new_tmp_pos

                            w[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = tmp_w
                            index = index + 1
                        else:
                            index = index + 2
                    else:
                        index = index + 1






def quantize(_model, width, _mode = None, _method_args = None, _shifting = None, _non_uniform = None, _var = 5): # _method_args = base
    '''
    Quantize the model with:
    _model       - neural network models
    width       - number of bits
    _is_uniform  - decide if we do uniform quantization
    _method      - if not uniform, assign quantization method: 'exp', or others (support in the future)
    _method_args - the parameters using for method-assigned quantization
    shifting     - if there is resistance shifting, it would be the value of variation
    '''

    original_weight = {}
    weight_quantized = []
    # print ('Quantization error:')
    for (ind, p) in enumerate(_model.parameters()):
        if len(p.data.size()) != 1:
            w = p.data

            # mask = (w != 0).float()
            w_pos = (w>0).float()*w
            w_neg = (w<0).float()*w*(-1)

            if (_mode == 'linear'):
                # get the quantization step

                start = w_pos.min()
                end = w_pos.max()
                q_step = (end-start)/(2**width-1)
                w_pos = ((w_pos - start)/q_step).round()*q_step + start


                original_weight[ind] = torch.zeros(w_pos.shape).cuda()
                original_weight[ind] += w_pos
###                if _shifting:
###                    print('Dynamic Variation: {}%'.format(1/_var))
###                    variation = np.random.normal(0, w_pos.cpu().numpy()/_var, w_pos.shape)
###                    w_pos -= torch.FloatTensor(variation).abs().cuda()
###                    if _non_uniform:
###                        non_uniform = np.random.normal(loc=_shifting, scale=0.1, size=w_pos.shape)
###                        non_uniform = torch.FloatTensor(non_uniform).cuda()
###                        w_pos = w_pos * non_uniform
###                    else:
###                        w_pos = w_pos * _shifting

                start = w_neg.min()

                q_step = (end-start)/(2**width-1)
                w_neg = ((w_neg - start)/q_step).round()*q_step + start

                original_weight[ind] += (-1)*w_neg
###                if _shifting:
##                    variation = np.random.normal(0, w_neg.cpu().numpy()/_var, w_neg.shape)
##                    w_neg -= torch.FloatTensor(variation).abs().cuda()
###                    if _non_uniform:
###                        non_uniform = np.random.normal(loc=_shifting, scale=0.1, size=w_neg.shape)
###                        non_uniform = torch.FloatTensor(non_uniform).cuda()
###                        w_neg = w_neg * non_uniform
###                    else:
###                        w_neg = w_neg * _shifting

                w = (w_pos + (-1)*w_neg)
                # print ('Layer - ', ind, ': ', (p.data-w).abs().mean())
                p.data = w
            elif (_mode == 'exp'):
                assert (_method_args is not None)
                base = _method_args # get the base
                shape = w_pos.shape

                # for positive part
                end = w_pos.max()
                scale = (end/(base**(2**width - 1))).cuda()                # get the largest value and scale  this,scale is a potion , and every conductance multiply this portion. it will be a according we
                step = (base**(torch.Tensor(range(2**width)))*scale).cuda()   # calc quantization intervals

#                assert (end == step[2**width - 1])

                masks = torch.zeros([2**width] + list(shape)).cuda() #zeros have to duplicate many pieces and it's size is list(shape) for each
                left = step[0]/2
                for k in range(2**width - 1):
                    right = (step[k] + step[k+1])/2
                    masks[k] = ((w_pos>left) & (w_pos<right)).float()
                    left = right
                masks[2**width - 1] = (w_pos>left).float() # for last interval

                w_pos_new = torch.zeros(shape).cuda()
                for k in range(2**width):
                    w_pos_new += masks[k].cuda() * step[k].cuda()

                original_weight[ind] = torch.zeros(w_pos_new.shape).cuda()
                original_weight[ind] += w_pos_new
####                if _shifting:
####                    print('Dynamic Variation: {}%'.format(1/_var))
####                    variation = np.random.normal(0, w_pos_new.cpu().numpy()/_var, w_pos_new.shape)
####                    w_pos_new -= torch.FloatTensor(variation).abs().cuda()
###                    if _non_uniform:
###                        non_uniform = np.random.normal(loc=_shifting, scale=0.1, size=w_pos_new.shape)
###                        non_uniform = torch.FloatTensor(non_uniform).cuda()
###                        w_pos_new = w_pos_new * non_uniform
###                    else:
###                        w_pos_new = w_pos_new * _shifting



                # for negative part
                end = w_neg.max()
                scale = (end/(base**(2**width - 1))).cuda()           # get the largest value and scale
                step = (base**(torch.Tensor(range(2**width)))*scale).cuda()  # calc quantization intervals
#                assert (end == step[2**width - 1])

                masks = torch.zeros([2**width] + list(shape)).cuda()
                left = step[0]/2
                for k in range(2**width - 1):
                    right = (step[k] + step[k+1])/2
                    masks[k] = ((w_neg>left) & (w_neg<right)).float()
                    left = right
                masks[2**width - 1] = (w_neg>left).float()

                w_neg_new = torch.zeros(shape).cuda()
                for k in range(2**width):
                    w_neg_new += masks[k]* step[k]

                original_weight[ind] += (-1)*w_neg_new
####                if _shifting:
####                    variation = np.random.normal(0, w_neg_new.cpu().numpy()/_var, w_neg_new.shape)
####                    w_neg_new -= torch.FloatTensor(variation).abs().cuda()
###                    if _non_uniform:
###                        non_uniform = np.random.normal(loc=_shifting, scale=0.1, size=w_neg_new.shape)
###                        non_uniform = torch.FloatTensor(non_uniform).cuda()
###                        w_neg_new = w_neg_new * non_uniform
###                    else:
###                        w_neg_new = w_neg_new * _shifting

                w = (w_pos_new + (-1)*w_neg_new)
                # print ('Layer - ', ind, ': ', (p.data-w).abs().mean())
                p.data = w

            elif (_mode == 'power'):
                assert (_method_args is not None)
                expo = _method_args # get the expo
                shape = w_pos.shape

                # for positive part
                end = w_pos.max()
                scale = (end/((2**width)**expo)).cuda()                       # get the largest value and scale
                step = ((torch.Tensor(range(1, 2**width + 1)))**expo*scale).cuda()   # calc quantization intervals
#                assert (end == step[2**width - 1])

                masks = torch.zeros([2**width] + list(shape)).cuda()
                left = step[0]/2
                for k in range(2**width - 1):
                    right = (step[k] + step[k+1])/2
                    masks[k] = ((w_pos>left) & (w_pos<right)).float()
                    left = right
                masks[2**width - 1] = (w_pos>left).float()

                w_pos_new = torch.zeros(shape).cuda()
                for k in range(2**width):
                    w_pos_new += masks[k].cuda() * step[k].cuda()

                original_weight[ind] = torch.zeros(w_pos_new.shape).cuda()
                original_weight[ind] += w_pos_new
###                if _shifting:
###                    if _non_uniform:
###                        non_uniform = np.random.normal(loc=_shifting, scale=0.1, size=w_pos_new.shape)
###                        non_uniform = torch.FloatTensor(non_uniform).cuda()
###                        w_pos_new = w_pos_new * non_uniform
###                    else:
###                        w_pos_new = w_pos_new * _shifting

                # for negative part
                end = w_neg.max()
                scale = (end/((2**width)**expo)).cuda()           # get the largest value and scale
                step = ((torch.Tensor(range(1, 2**width + 1)))**expo*scale).cuda()  # calc quantization intervals
#                assert (end == step[2**width - 1])

                masks = torch.zeros([2**width] + list(shape)).cuda()
                left = step[0]/2
                for k in range(2**width - 1):
                    right = (step[k] + step[k+1])/2
                    masks[k] = ((w_neg>left) & (w_neg<right)).float()
                    left = right
                masks[2**width - 1] = (w_neg>left).float()

                w_neg_new = torch.zeros(shape).cuda()
                for k in range(2**width):
                    w_neg_new += masks[k]* step[k]

                original_weight[ind] += (-1)*w_neg_new
###                if _shifting:
###                    if _non_uniform:
###                        non_uniform = np.random.normal(loc=_shifting, scale=0.1, size=w_neg_new.shape)
###                        non_uniform = torch.FloatTensor(non_uniform).cuda()
###                        w_neg_new = w_neg_new * non_uniform
###                    else:
###                        w_neg_new = w_neg_new * _shifting

                w = (w_pos_new + (-1)*w_neg_new)
                # print ('Layer - ', ind, ': ', (p.data-w).abs().mean())
                p.data = w

    return original_weight

def SaveWeights(_original_weight, _gradient_sum, _model):
    N = 100
    total = 0
    total_p = 0
    ratio = 0.1
    for ind, p in enumerate(_model.parameters()):
        if len(p.data.size()) != 1:
            ###print(_gradient_sum[ind].shape)
            ###print(_gradient_sum[ind].view(-1).shape[0])

            _gradient_sum[ind] = _gradient_sum[ind].view(-1)
###            N = int(_gradient_sum[ind].shape[0] * ratio)

            N = _gradient_sum[ind].shape[0]
            if N > 1000:
                N = int(N * ratio)

            total += N
            total_p += _gradient_sum[ind].shape[0]
            abs_gradient_sum = _gradient_sum[ind].abs()
            top_k, index = torch.topk(abs_gradient_sum, N)

            shape = p.data.shape
            p.data = p.data.view(-1)
            p.data[index] = _original_weight[ind].view(-1)[index]
            p.data = p.data.view(shape)
    print("Total:", total)
    print("Total_p:", total_p)
    print("Saving percentage: %.2f %%" % (100*float(total)/total_p))



def RowRestoring_Weights(_original_weight, _model, _rate):
    ratio = _rate
    row = 128
    col = 128
    total = 0
    total_p = 0

    for ind, p in enumerate(_model.parameters()):
        if len(p.data.size()) != 1:
            ###print(_gradient_sum[ind].shape)
            ###print(_gradient_sum[ind].view(-1).shape[0])

            total_p += p.data.view(-1).shape[0]
            shape = p.data.shape
            w = p.data.view(shape[0], -1)
            _original_weight[ind] = _original_weight[ind].view(shape[0], -1)
            w_pos = ((w>0).float()*w).abs()
            w_neg = ((w<0).float()*w).abs()
            ori_w_pos = ((w>0).float() * _original_weight[ind]).abs()
            ori_w_neg = ((w<0).float() * _original_weight[ind]).abs()

            w_row_size = w.shape[0]
            w_col_size = w.shape[1]
            NumRow = math.ceil(w_row_size / row)
            NumCol = math.ceil(w_col_size / col)
            rram_pos = {}
            rram_neg = {}
            rram_pos_sum_row = {}
            rram_neg_sum_row = {}

            for i in range(NumRow):
                for j in range(NumCol):
                    rram_pos[i,j] = w_pos[i*row:(i+1)*row, j*col:(j+1)*col]
                    rram_neg[i,j] = w_neg[i*row:(i+1)*row, j*col:(j+1)*col]
                    rram_pos_sum_row[i,j] = torch.sum(rram_pos[i,j], dim=1)
                    rram_neg_sum_row[i,j] = torch.sum(rram_neg[i,j], dim=1)
                    size = math.ceil(rram_pos_sum_row[i,j].shape[0] * ratio)
                    top_k_pos, index_pos = torch.topk(rram_pos_sum_row[i,j], size)
                    top_k_neg, index_neg = torch.topk(rram_neg_sum_row[i,j], size)
                    w_pos[i*row+index_pos, j*col:(j+1)*col] = ori_w_pos[i*row+index_pos, j*col:(j+1)*col]
                    w_neg[i*row+index_neg, j*col:(j+1)*col] = ori_w_neg[i*row+index_neg, j*col:(j+1)*col]
                    total += rram_pos[i,j].shape[1] * size

            w = (w_pos + (-1)*w_neg)
            w = w.view(shape)
            p.data = w

    print("Total:", total)
    print("Total_p:", total_p)
    print("Saving percentage: %.2f %%" % (100*float(total)/total_p))



def RowRestoring_Gradients(_original_weight, _gradient_sum, _model, _rate):
    ratio = _rate
    row = 128
    col = 128
    total = 0
    total_p = 0

    for ind, p in enumerate(_model.parameters()):
        if len(p.data.size()) != 1:
            ###print(_gradient_sum[ind].shape)
            ###print(_gradient_sum[ind].view(-1).shape[0])

            total_p += p.data.view(-1).shape[0]
            shape = p.data.shape
            w = p.data.view(shape[0], -1)
            _original_weight[ind] = _original_weight[ind].view(shape[0], -1)
            _gradient_sum[ind] = _gradient_sum[ind].view(shape[0], -1)
            w_pos = (w>0).float()*w
            w_neg = (w<0).float()*w
            ori_w_pos = (w>0).float() * _original_weight[ind]
            ori_w_neg = (w<0).float() * _original_weight[ind]
            g_pos = ((w>0).float() * _gradient_sum[ind]).abs()
            g_neg = ((w<0).float() * _gradient_sum[ind]).abs()

            grad_row_size = _gradient_sum[ind].shape[0]
            grad_col_size = _gradient_sum[ind].shape[1]
            NumRow = math.ceil(grad_row_size / row)
            NumCol = math.ceil(grad_col_size / col)
            rram_pos = {}
            rram_neg = {}
            rram_pos_sum_row = {}
            rram_neg_sum_row = {}

            for i in range(NumRow):
                for j in range(NumCol):
                    rram_pos[i,j] = g_pos[i*row:(i+1)*row, j*col:(j+1)*col]
                    rram_neg[i,j] = g_neg[i*row:(i+1)*row, j*col:(j+1)*col]
                    rram_pos_sum_row[i,j] = torch.sum(rram_pos[i,j], dim=1)
                    rram_neg_sum_row[i,j] = torch.sum(rram_neg[i,j], dim=1)
                    size = math.ceil(rram_pos_sum_row[i,j].shape[0] * ratio)
                    top_k_pos, index_pos = torch.topk(rram_pos_sum_row[i,j], size)
                    top_k_neg, index_neg = torch.topk(rram_neg_sum_row[i,j], size)
                    w_pos[i*row+index_pos, j*col:(j+1)*col] = ori_w_pos[i*row+index_pos, j*col:(j+1)*col]
                    w_neg[i*row+index_neg, j*col:(j+1)*col] = ori_w_neg[i*row+index_neg, j*col:(j+1)*col]
                    total += rram_pos[i,j].shape[1] * size

            w = w_pos + w_neg
            w = w.view(shape)
            p.data = w

    print("Total:", total)
    print("Total_p:", total_p)
    print("Saving percentage: %.2f %%" % (100*float(total)/total_p))



#def Train(trainloader, model, criterion, optimizer, epoch, qtz_opts, shifting, non_uniform, var, args, prune_ratio_layer): ## for prune layer only
def Train(trainloader, model, criterion, optimizer, epoch, qtz_opts, shifting, non_uniform, var, args): ## for prune layer only
#def Train(trainloader, model, criterion, optimizer, epoch, qtz_opts, shifting, non_uniform, var, args): ## original train

    model.train()

    correct = 0
    total = 0

    iterations = 2
    for i, (x, y) in enumerate(trainloader):

        x = Variable(x).cuda()
        y = Variable(y).cuda()
        if qtz_opts.is_quantize and (i % iterations == 0):
            original_weight = quantize(model, width = qtz_opts.width, _mode = qtz_opts.mode, _method_args = qtz_opts.base, _shifting = shifting, _non_uniform = non_uniform, _var = var)

        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        if args.stop_gradient: ### stop update gradient
            for name, param in (model.named_parameters()):
                #print(name)
                if len(param.data.size()) == 4 :
                    row = param.data.shape[1]*param.data.shape[2]*param.data.shape[3]
                    numRow = math.ceil(row/128)
                    numCol = math.ceil(param.data.shape[0]/128)
                    rram_partition = {}
                    rram_partition_total_sum = {}
                    rram_partition_total_sum_list=[]
                    SubarrayCol = 128
                    SubarrayRow = 128
                    #print(name)
                    if name != 'module.layer1.0.conv2.weight' and name != 'module.layer1.1.conv2.weight' and name !='module.layer1.2.conv2.weight' and name != 'module.layer2.0.conv2.weight' and name != 'module.layer2.1.conv2.weight' and name != 'module.layer2.2.conv2.weight' and name != 'module.layer2.3.conv2.weight' and name != 'module.layer3.0.conv2.weight' and name != 'module.layer3.1.conv2.weight' and name != 'module.layer3.2.conv2.weight' and name != 'module.layer3.3.conv2.weight' and name != 'module.layer3.4.conv2.weight' and name != 'module.layer3.5.conv2.weight' and name != 'module.layer4.0.conv2.weight' and name != 'module.layer4.1.conv2.weight' and name != 'module.layer4.2.conv2.weight':
                        g = param.grad.reshape(param.grad.shape[0],row)
                        w = param.grad.reshape(param.data.shape[0],row)

                    for i in range(numCol):
                        for j in range(numRow):
                            rram_partition[i,j] = w[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                            rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))

                    for key, value in rram_partition_total_sum.items():
                        tmp = [key,value]
                        rram_partition_total_sum_list.append(tmp)

                    rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])

                    ############# vgg8 ############
                    if args.arch == 'vgg8':
                        #ratio = int(args.prune_ratio)/100
                        if name  == 'features.module.2.weight':
                            ratio = prune_ratio_layer[1]
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(128,128)

                        if name  == 'features.module.5.weight':
                            ratio = prune_ratio_layer[2]
                            #ratio = 0.2
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(128,128)

                        if name  == 'features.module.7.weight':
                            ratio = prune_ratio_layer[3]
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(128,128)

                        if name  == 'features.module.10.weight':
                            ratio = prune_ratio_layer[4]
                            #ratio = 0.3
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(128,128)

                        if name  == 'features.module.12.weight':
                            ratio = prune_ratio_layer[5]
                            #ratio = 0.3
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(128,128)

                        param.grad = g.reshape(param.grad.shape)



                    if args.arch == 'vgg11':
                        ratio= int(args.prune_ratio)/100
                        if name  == 'module.features.module.3.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)

                        if name  == 'mmodule.features.module.6.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)

                        if name  == 'module.features.module.8.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                        if name  == 'module.features.module.11.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                        if name  == 'module.features.module.13.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                        if name  == 'module.features.module.16.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)

                        if name  == 'module.features.module.18.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)

                        param.grad = g.reshape(param.grad.shape)

                    if args.arch == 'alexnet':
                        ratio = int(args.prune_ratio)/100
                        if name  == 'module.features.3.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                        if name  == 'module.features.6.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                        if name  == 'module.features.8.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                        if name  == 'module.features.10.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                        param.grad = g.reshape(param.grad.shape)

                    if args.arch == 'resnet34':
                        ratio = int(args.prune_ratio)/100
                        if name == 'module.layer1.0.conv1.weight' or name =='module.layer1.1.conv1.weight' or name == 'module.layer1.2.conv1.weight' or name =='module.layer2.0.conv1.weight' or name =='module.layer2.1.conv1.weight' or name == 'module.layer2.2.conv1.weight' or name == 'module.layer2.3.conv1.weight' or name == 'module.layer3.0.conv1.weight' or name == 'module.layer3.0.shortcut.0.weight' or name == 'module.layer3.1.conv1.weight' or name == 'module.layer3.2.conv1.weight' or name == 'module.layer3.3.conv1.weight' or name == 'module.layer3.4.conv1.weight' or name == 'module.layer3.5.conv1.weight' or name == 'module.layer4.0.conv1.weight' or name == 'module.layer4.0.shortcut.0.weight' or name == 'module.layer4.1.conv1.weight' or name == 'module.layer4.2.conv1.weight' :

                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)

                            param.grad = g.reshape(param.grad.shape)




                if len(param.data.size()) == 2:
                    row = param.data.shape[1]
                    numRow = math.ceil(row/128)
                    numCol = math.ceil(param.data.shape[0]/128)
                    rram_partition = {}
                    rram_partition_total_sum = {}
                    rram_partition_total_sum_list = []
                    SubarrayCol = 128
                    SubarrayRow = 128
                    g = param.grad.reshape(param.grad.shape[0],row)

                    for i in range(numCol):
                        for j in range(numRow):
                            rram_partition[i,j] = g[i*SubarrayCol:(i+1)*SubarrayCol, j*SubarrayRow:(j+1)*SubarrayRow]
                            rram_partition_total_sum[i,j] = torch.sum(torch.abs(rram_partition[i,j]))

                    for key, value in rram_partition_total_sum.items():
                        tmp = [key,value]
                        rram_partition_total_sum_list.append(tmp)

                    rram_partition_total_sum_list = sorted(rram_partition_total_sum_list, key = lambda s:s[1])

                    ####vgg8#######
                    if args.arch =='vgg8':
                        #ratio = int(args.prune_ratio)/100
                        if name  == 'classifier.0.weight':
                            ratio = prune_ratio_layer[6]
                            #ratio = 0.9
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(128,128)

                        if name  == 'classifier.2.weight':
                            ratio = prune_ratio_layer[7]
                            #ratio = 0
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(10,128)
                        param.grad = g.reshape(param.grad.shape)
                    ##### vgg11 ###
                    if args.arch == 'vgg11' :
                        ratio= int(args.prune_ratio)/100
                        if name  == 'module.classifier.0.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                        if name  == 'module.classifier.2.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                        if name  == 'module.classifier.4.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                        param.grad = g.reshape(param.grad.shape)

                    if args.arch == 'alexnet':
                        ratio = int(args.prune_ratio)/100
                        if name  == 'module.classifier.0.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                        if name  == 'module.classifier.2.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                        if name  == 'modukle.classifier.4.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                        param.grad = g.reshape(param.grad.shape)
                    if args.arch == 'resnet34':
                        ratio = int(args.prune_ratio)/100
                        if name  == 'module.linear.weight':
                            #ratio = 0.4
                            pruned_subarray = math.ceil(len(rram_partition_total_sum) * ratio)
                            for i in range(pruned_subarray):
                                key,value = rram_partition_total_sum_list[i]
                                a=g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow]
                                g[key[0]*SubarrayCol:(key[0]+1)*SubarrayCol, key[1]*SubarrayRow:(key[1]+1)*SubarrayRow] = torch.zeros(a.shape)
                        param.grad = g.reshape(param.grad.shape)



        optimizer.step()

        _, pred = torch.max(output.data, 1)
        total += y.size(0)
        correct +=  (pred == y.data).sum().cpu()

    print('(Epoch - %d) Accuracy on Trainset: %.2f %%' % (epoch, 100*float(correct)/total))

#def Eval(testloader, model, epoch, qtz_opts, shifting, non_uniform):
def Eval(testloader, model, criterion, optimizer, epoch, qtz_opts, shifting, non_uniform, rate, mode, var):

    model.eval()
    correct = 0
    correct_new = 0
    total = 0
    total_new = 0
    first_time = True
    gradient_sum = {}
    if qtz_opts.is_quantize:
        original_weight = quantize(model, width = qtz_opts.width, _mode = qtz_opts.mode, _method_args = qtz_opts.base, _shifting = shifting, _non_uniform = non_uniform, _var = var)
        if shifting:
            ##
            print('in eval-quantize-shifting')
            ##
            DynamicVariation(model, var)




    for ind,(x, target) in enumerate(testloader):

        x = Variable(x).cuda()
        target = Variable(target).cuda()
        output = model(x)




        ##Sum the total gradients
        if shifting:
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()



            for ind, p in enumerate(model.parameters()):
                if len(p.data.size()) != 1:
                    grad = p.grad
                    if first_time:
                        gradient_sum[ind] = torch.zeros(grad.shape).cuda()
                    gradient_sum[ind] += grad
            first_time = False

        _, pred = torch.max(output.data, 1)
        total += target.size(0)
        correct +=  (pred == target.data).sum().cpu()

    if shifting:
        assert (mode == 'w' or mode == 'g')
        if mode == 'w':
            print('Mode: Weights')
            RowRestoring_Weights(original_weight, model, rate)
        else:
            print('Mode: Gradients')
            RowRestoring_Gradients(original_weight, gradient_sum, model, rate)
#        SaveWeights(original_weight, gradient_sum, model)
        for ind,(x, target) in enumerate(testloader):
            x = Variable(x).cuda()
            target = Variable(target).cuda()
            output = model(x)

            _, pred = torch.max(output.data, 1)
            total_new += target.size(0)
            correct_new +=  (pred == target.data).sum().cpu()

    if epoch != -1:
        print('(Epoch - %d) Accuracy on Testset: %.2f %%' % (epoch, 100*float(correct)/total))
    else:
        print('Accuracy on Testset: %.2f %%' % (100*float(correct)/total))
        if shifting:
            print('Accuracy on Testset after saving: %.2f %%' % (100*float(correct_new)/total_new))


    return (float(correct)/total)

