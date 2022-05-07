import torch.nn as nn
from torch.utils.data import Dataset

from openselfsup.utils import nondist_forward_collect, dist_forward_collect

from operator import itemgetter
class Extractor(object):
    """Feature extractor.

    Args:
        dataset (Dataset | dict): A PyTorch dataset or dict that indicates
            the dataset.
        imgs_per_gpu (int): Number of images on each GPU, i.e., batch size of
            each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        dist_mode (bool): Use distributed extraction or not. Default: False.
        return_gt_label(bool): Return ground truth label or not. Default: False.
    """

    def __init__(self,
                 dataset,
                 imgs_per_gpu,
                 workers_per_gpu,
                 dist_mode=False,
                 return_gt_label=False):
        from openselfsup import datasets
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.data_loader = datasets.build_dataloader(
            self.dataset,
            imgs_per_gpu,
            workers_per_gpu,
            dist=dist_mode,
            shuffle=False)
        self.dist_mode = dist_mode
        self.return_gt_label = return_gt_label
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _forward_func(self, runner, **x):
        backbone_feat = runner.model(mode='extract', **x)
        #### resnet backbone:
        last_layer_feat = runner.model.module.neck([backbone_feat[-1]])[0]
        #### WRN backbone:
        #last_layer_feat = runner.model.module.neck([backbone_feat[0]])[0]
        last_layer_feat = last_layer_feat.view(last_layer_feat.size(0), -1) 
        if self.return_gt_label:
            return dict(feature=last_layer_feat.cpu(),gt_label=x['gt_label'])
        else:
            return dict(feature=last_layer_feat.cpu())

    def __call__(self, runner):
        func = lambda **x: self._forward_func(runner, **x)
        if self.dist_mode:
            if self.return_gt_label:
                ret_tuple = itemgetter('feature','gt_label')(dist_forward_collect(
                    func,
                    self.data_loader,
                    runner.rank,
                    len(self.dataset),
                    ret_rank=-1))  # NxD
                feats = ret_tuple[0]
                gt_labels = ret_tuple[1]
                
            else:
                feats = dist_forward_collect(
                    func,
                    self.data_loader,
                    runner.rank,
                    len(self.dataset),
                    ret_rank=-1)['feature']  # NxD
              
        else:
            feats = nondist_forward_collect(func, self.data_loader,
                                            len(self.dataset))['feature']
        if self.return_gt_label:
            return feats, gt_labels
        else:
            return feats
