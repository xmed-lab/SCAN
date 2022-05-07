import numpy as np
import torch
import torch.nn as nn

from openselfsup.utils import print_log
from . import builder
from .registry import MODELS
from .utils import Sobel

from tensorboard_logger import Logger
from datetime import datetime

now = datetime.now()

@MODELS.register_module
class SCAN(nn.Module):
    """SCAN: Sub-cluster Aware Network
    Losses: 1. Supervised cross entropy loss, 2. odc loss, 3. purity losses
    Changing based on:
    Official implementation of
    "Online Deep Clustering for Unsupervised Representation Learning
    (https://arxiv.org/abs/2006.10645)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        with_sobel (bool): Whether to apply a Sobel filter on images. Default: False.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        memory_bank (dict): Module of memory banks. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self,
                 backbone,
                 with_sobel=False,
                 neck=None,
                 head=None,
                 memory_bank=None,
                 pretrained=None):
        super(SCAN, self).__init__()
        self.with_sobel = with_sobel
        if with_sobel:
            self.sobel_layer = Sobel()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        if head is not None:
            self.head = builder.build_head(head)
        if memory_bank is not None:
            self.memory_bank = builder.build_memory(memory_bank)
        self.init_weights(pretrained=pretrained)

        # set reweight tensors
        self.num_clusters = memory_bank.num_clusters
        self.loss_weight = torch.ones((self.num_clusters, ),
                                      dtype=torch.float32).cuda()
        self.loss_weight /= self.loss_weight.sum()

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights(init_linear='kaiming')
        self.head.init_weights(init_linear='normal')

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        if self.with_sobel:
            img = self.sobel_layer(img)
        x = self.backbone(img)
        return x

    def forward_train(self, img, idx, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            idx (Tensor): Index corresponding to each image.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # forward & backward
        x = self.forward_backbone(img)
        feature = self.neck(x)
        outs = self.head(feature)
        positives = self.memory_bank.class_centroids[kwargs['gt_label']]
        cluster_idx = self.memory_bank.cluster_label_bank[idx]
        negatives = []
        for i, feat in enumerate(feature[0]):
            cluster_members_idx = [j for j, (e,f) in enumerate(zip(self.memory_bank.cluster_label_bank,self.memory_bank.class_label_bank)) if e == cluster_idx[i] and f != kwargs['gt_label'][i]]
            if not cluster_members_idx:
                # deal with purity cluster
                cluster_members_idx = [j for j, f in enumerate(self.memory_bank.class_label_bank) if f != kwargs['gt_label'][i]]
            
            # the features of those samples
            cluster_members_feats = self.memory_bank.feature_bank[cluster_members_idx].cuda()
            # the euclidean distance between feature[i] and those features
            distance_i = torch.cdist(feat.unsqueeze(0), cluster_members_feats, p=2).squeeze()
            # the indx of sample with minimum distance
            negative_idx = cluster_members_idx[torch.argmin(distance_i)]
            # append the feature of this sample
            negatives.append(self.memory_bank.feature_bank[negative_idx])
        negatives = torch.stack(negatives, dim=0)
        if self.memory_bank.cluster_label_bank.is_cuda:
            loss_inputs = (outs[0], kwargs['gt_label'], outs[1], self.memory_bank.cluster_label_bank[idx], feature[0], positives, negatives)
        else:
            loss_inputs = (outs[0], kwargs['gt_label'].cuda(),outs[1], self.memory_bank.cluster_label_bank[idx.cpu()].cuda(), feature[0].cuda(), positives.cuda(), negatives.cuda())
        losses = self.head.loss(*loss_inputs)

        # update samples memory
        change_ratio = self.memory_bank.update_samples_memory(
            idx, feature[0].detach())
        losses['change_ratio'] = change_ratio

        return losses

    def forward_test(self, img, **kwargs):
        x = self.forward_backbone(img)  # tuple
        feature = self.neck(x)
        outs = self.head(feature)
        keys = ['head0']
        out_tensors = [outs[0].cpu()]
        return dict(zip(keys, out_tensors))

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))

    def set_reweight(self, labels=None, reweight_pow=0.5):
        """Loss re-weighting.

        Re-weighting the loss according to the number of samples in each class.

        Args:
            labels (numpy.ndarray): Label assignments. Default: None.
            reweight_pow (float): The power of re-weighting. Default: 0.5.
        """
        if labels is None:
            if self.memory_bank.cluster_label_bank.is_cuda:
                labels = self.memory_bank.cluster_label_bank.cpu().numpy()
            else:
                labels = self.memory_bank.cluster_label_bank.numpy()
        hist = np.bincount(
            labels, minlength=self.num_clusters).astype(np.float32)
        inv_hist = (1. / (hist + 1e-5))**reweight_pow
        weight = inv_hist / inv_hist.sum()
        self.loss_weight.copy_(torch.from_numpy(weight))
        self.head.criterion2 = nn.CrossEntropyLoss(weight=self.loss_weight)
