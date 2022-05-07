# SCAN: Sub-Cluster-Aware Network for Few-shot Skin Disease Classification
paper link: TBA

Based on a crucial observation that skin disease images often exist multiple sub-clusters within a class (i.e., the appearances of images within one class of disease vary and form multiple distinct sub-groups), we design a novel Sub-Cluster- Aware Network, namely SCAN, for rare skin disease diagnosis with enhanced accuracy. 

## Install

We biult this program based on OpenMMLab Self-Supervised Learning Toolbox. You need to download and install related packages as bellow:
- Create conda environment and activate it:
```
conda create -n scan python=3.7
conda activate scan
```
- Check CUDA version:
```
nvcc --version
```

- Install Pytorch, please follow [official instructions](https://pytorch.org/).
E.g.:
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```

- Check pytorch version:
```
python
>>> import torch
>>> torch.__version__
>>> quit()
```
- Install MMCV:
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
You need to replace {cu_version} and {torch_version} according to your own device.  
E.g.: my CUDA version: 10.3; my torch version: 1.7:
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7/index.html
```

- Clone repository and install:
```
git clone https://github.com/xmed-lab/SCAN.git
cd SCAN/openselfsup
pip install -v -e .
```

## Usage

### Prepare data
We use SD-198 and Derm7pt in our paper. These two public datasets can be downloaded here:  
- SD-198: http://xiaopingwu.cn/assets/projects/sd-198/
- Derm7pt: https://derm.cs.sfu.ca/Download.html  

You need to move all the images and files into "SCAN/SkinLesionData/SD-198-20/" and "SCAN/SkinLesionData/Derm7pt/" respectively, and run split.py file to genenrate json files. You need to change the corresponding dataset path in split.py files.

### Train 

We provided several config files in "SCAN/openselfsup/configs/scan", you can either use them or create new files for your own experiments.  
To train SCAN model on SD-198 dataset with WRN as backbone, e.g.:
```
cd SCAN/openselfsup
CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/scan/sd198-20/wrn_v1.py 1
```
The path files are saved in "SCAN/openselfsup/work_dirs/scan/" directory.

### Save feature

After training, you need to extract and save features from base and novel set for later evaluation, e.g.:
```
cd SCAN/evaluate
mkdir checkpoints
CUDA_VISIBLE_DEVICES=0 python save_plk.py --dataset SD-198-20 --method scan --model WideResNet28_10 --modelfile ../openselfsup/work_dirs/scan/sd198-20/wrn_v1/epoch_800.pth
```

### Evaluate

You need to edit 'SCAN/evaluate/FSLTask.py' file to add your feature file. In line 9 of 'SCAN/evaluate/FSLTask.py' file, please add dict object to '_datasetFeaturesFiles' according to your path.  
E.g.:
```
"sd198-20_wrn_v1": "./checkpoints/SD-198-20/WideResNet28_10_scan/last/novel_features.plk",
 ```


Before evaluating on novel set, you need to create empty 'cache' directory.  
For example, evaluating for 2-way 1-shot setting:
```
cd SCAN/evaluation
mkdir cache
python evaluate_DC_modified.py --dataset SD-198-20 --model WideResNet28_10 --method scan --dataset_file sd198-20_wrn_v1 --n_shot 1 --n_way 2
```

## Citation

If our paper is useful for your research, please cite our paper

## Implement references

[MMSelfsup](https://github.com/open-mmlab/mmselfsup)

[A Closer Look at Few-shot Classification](https://github.com/wyharveychen/CloserLookFewShot)

[Free Lunch for Few-Shot Learning: Distribution Calibration](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration)