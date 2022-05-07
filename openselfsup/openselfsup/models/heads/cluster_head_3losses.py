import torch.nn as nn
import torch
from mmcv.cnn import kaiming_init, normal_init

from ..utils import accuracy
from ..registry import HEADS


@HEADS.register_module
class ClusterHead3Losses(nn.Module):
    """Cluster classifier head, with two heads: cluster and classifier head
    three losses: class loss, cluster loss, purity loss
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 num_classes=1000,
                 num_clusters=10000,
                 weight_ce=1.0,
                 weight_odc=1.0,
                 weight_p=1.0):
        super(ClusterHead3Losses, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        
        #self.weight2 = torch.ones(50)
        self.weight_ce = weight_ce
        self.weight_odc = weight_odc
        self.weight_p = weight_p
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.CrossEntropyLoss()
        self.criterion3 = nn.TripletMarginLoss(margin=0.3, p=2)
        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls1 = nn.Linear(in_channels, num_classes)
        self.fc_cls2 = nn.Linear(in_channels, num_clusters)

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        #self.weight = None
        assert init_linear in ['normal', 'kaiming'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            assert x.dim() == 4, \
                "Tensor must has 4 dims, got: {}".format(x.dim())
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = [fc(x) for fc in [self.fc_cls1, self.fc_cls2]]
        
        return x

    def loss(self, class_score, gt_labels, cluster_score, pseudo_labels, anchor, positive, negative):
        losses = dict()
        #self.weight = 'None'
        
        losses['loss.{}'.format('_class')] = self.weight_ce * self.criterion1(class_score, gt_labels)
        losses['acc.{}'.format('_class')] = accuracy(class_score, gt_labels)
        losses['loss.{}'.format('_cluster')] = self.weight_odc * self.criterion2(cluster_score, pseudo_labels)
        losses['acc.{}'.format('_cluster')] = accuracy(cluster_score, pseudo_labels)
        losses['loss.{}'.format('_purity')] = self.weight_p * self.criterion3(anchor, positive, negative)

        return losses

    