import fnmatch
import numpy as np
import os
from os import listdir
from os.path import isfile, isdir, join
import json
import random
## Change to your own path ##
data_path = '/home/slidm/FSL/SCAN/SkinLesionData/SD-198-20/images' 

test_class_list = []
train_class_list = []
for subdir, dirs, files in os.walk(data_path):
    # subdir: "./images/Acne_Keloidalis_Nuchae"
    number = len(fnmatch.filter(os.listdir(subdir), '*.jpg'))
    if number < 20 and number > 0:
        test_class_list.append(subdir)
    train_class_list.append(subdir)


savedir = './'
dataset_list = ['base','val','novel']

#if not os.path.exists(savedir):
#    os.makedirs(savedir)

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
print(len(folder_list)) # ['Acne_Keloidalis_Nuchae', 'Acne_Vulgaris',...]


classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.') and cf[-4:] == ".jpg"])
    random.shuffle(classfile_list_all[i])

# generate json files

num_base = 0
num_val = 0
num_novel = 0
for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'base' in dataset:
            if (len(classfile_list)==60 and i%2 != 1):
                if (num_base<20):
                    num_base += 1
                    file_list = file_list + classfile_list
                    label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset:
            if (len(classfile_list)>=20 and i%8 == 1):
                num_val += 1
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'novel' in dataset:
            if (len(classfile_list)<20 and len(classfile_list)>0):
                num_novel += 1
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list])
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

print("base:", num_base, "val:", num_val, "novel:", num_novel)
