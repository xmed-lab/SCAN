import fnmatch
import numpy as np
import os
from os import listdir
from os.path import isfile, isdir, join
import json
import random
import csv
import pandas as pd
## Change to your own path ##
data_path = '/home/slidm/FSL/SCAN/SkinLesionData/Derm7pt/images/'

meta_file = './meta/meta.csv'
meta = pd.read_csv(meta_file)

cl_list = meta['diagnosis'].unique()

savedir = './'
dataset_list = ['base', 'novel']

#if not os.path.exists(savedir):
#    os.makedirs(savedir)

classfile_list_all = []

for i, cl_name in enumerate(cl_list):
    if cl_name not in ['melanoma', 'miscellaneous']:
        data = meta.loc[meta['diagnosis'] == cl_name]
        clinical_data_list = data['clinic'].tolist()
        derm_data_list = data['derm'].tolist()
        data_list = clinical_data_list + derm_data_list
        data_list = [data_path + data for data in data_list]
        if cl_name == 'melanoma metastasis':
            data_list.append(data_list[2])
            data_list.append(data_list[7])
        random.shuffle(data_list)
        classfile_list_all.append(data_list)
        


# generate json files


num_base = 0
num_val = 0
num_novel = 0
num_images = 0
for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'base' in dataset:
            if (len(classfile_list)>=40):
                num_base += 1
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
                num_images += len(classfile_list)
        if 'val' in dataset:
            if (len(classfile_list)>=20 and i%8 == 1):
                num_val += 1
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
                num_images += len(classfile_list)
        if 'novel' in dataset:
            if (len(classfile_list)<40 and len(classfile_list)>0):
                num_novel += 1
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
                num_images += len(classfile_list)

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in cl_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
print("base:", num_base, "novel:", num_novel)
