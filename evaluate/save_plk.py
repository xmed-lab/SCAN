from __future__ import print_function

import argparse
import csv
import os
import collections
import pickle
import random

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from io_utils import parse_args
from data.datamgr import SimpleDataManager , SetDataManager
import configs

import wrn_model

import torch.nn.functional as F

from io_utils import parse_args, get_resume_file ,get_assigned_file
from os import path

import backbone

use_gpu = torch.cuda.is_available()

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module 
    def forward(self, x):
        return self.module(x)

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def extract_feature(val_loader, model, checkpoint_dir, tag='last',set='base'):
    save_dir = '{}/{}'.format(checkpoint_dir, tag)
    if os.path.isfile(save_dir + '/%s_features.plk'%set):
        data = load_pickle(save_dir + '/%s_features.plk'%set)
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    #model.eval()
    with torch.no_grad():
        
        output_dict = collections.defaultdict(list)

        for i, (inputs, labels) in enumerate(val_loader):
            # compute output
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs,_ = model(inputs) # WRN
            #outputs = model(inputs) # ResNet
            print("outputs size:", outputs[0].size())
            outputs = outputs.cpu().data.numpy()

            for out, label in zip(outputs, labels):
                output_dict[label.item()].append(out)
            
        all_info = output_dict
        save_pickle(save_dir + '/%s_features.plk'%set, all_info)
        return all_info

if __name__ == '__main__':
    params = parse_args('test')

    loadfile_base = configs.data_dir[params.dataset] + 'base.json'
    loadfile_novel = configs.data_dir[params.dataset] + 'novel.json'
    datamgr       = SimpleDataManager(80, batch_size = 128)
    base_loader = datamgr.get_data_loader(loadfile_base, aug=False)
    novel_loader      = datamgr.get_data_loader(loadfile_novel, aug = False)
    
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    
    

    if params.model == 'WideResNet28_10':
        print("Using wrn_model.wrn28_10(num_classes) model...")
        model = wrn_model.wrn28_10(num_classes=params.num_classes)
        model_params = model.state_dict() 
        for k,v in model_params.items():
            print(k) 
    elif params.model == 'ResNet18':
        print("Using backbone.ResNet18 model...")
        model = backbone.ResNet18() 
        model_params = model.state_dict() 
        for k,v in model_params.items():
            print(k) 
    elif params.model == 'ResNet34':
        print("Using backbone.ResNet34 model...")
        model = backbone.ResNet34() 
        model_params = model.state_dict() 
        for k,v in model_params.items():
            print(k) 
    elif params.model == 'Conv4':
        print("Using backbone.Conv4() model...")
        model = backbone.Conv4()
        model_params = model.state_dict() 
        for k,v in model_params.items():
            print(k) 
    elif params.model == 'Conv6':
        print("Using backbone.Conv6() model...")
        model = backbone.Conv6()
        model_params = model.state_dict() 
        for k,v in model_params.items():
            print(k) 

    model = model.cuda()
    cudnn.benchmark = True

    checkpoint = torch.load(params.modelfile)
    print("KEYs:",checkpoint.keys())
    state = checkpoint['state_dict'] 
    state_keys = list(state.keys())
    print(state_keys)
    callwrap = False

    #if 'module' in state_keys[0]:
    #    callwrap = True
    #if callwrap:
    #    model = WrappedModel(model)


    for i, key in enumerate(state_keys):
        if "backbone." in key and not("neck." in key) and not("head." in key) :
            newkey = key.replace("backbone.","")  
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

    model_dict_load = model.state_dict()
    model_dict_load.update(state)
    model.load_state_dict(model_dict_load)
    model.eval()
    output_dict_base = extract_feature(base_loader, model, checkpoint_dir, tag='last', set='base')
    print("base set features saved!")
    output_dict_novel=extract_feature(novel_loader, model, checkpoint_dir, tag='last',set='novel')
    print("novel features saved!")


# CUDA_VISIBLE_DEVICES=3 python save_plk.py --dataset SD-198-20 --method scan --model WideResNet28_10 --modelfile ../openselfsup/work_dirs/scan/sd198-20/wrn_v1/epoch_800.pth
