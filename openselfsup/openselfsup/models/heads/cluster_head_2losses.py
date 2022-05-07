import torch.nn as nn
import torch
from mmcv.cnn import kaiming_init, normal_init

from ..utils import accuracy
from ..registry import HEADS


@HEADS.register_module
class ClusterHead2Losses(nn.Module):
    """Cluster classifier head, with two heads: cluster and classifier head
    two losses: class loss, cluster loss
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 num_classes=1000,
                 num_clusters=10000):
        super(ClusterHead2Losses, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.weight1 = torch.ones(200)
        #self.weight2 = torch.ones(50)
        self.criterion1 = nn.CrossEntropyLoss(weight=self.weight1)
        self.criterion2 = nn.CrossEntropyLoss()

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

    def loss(self, class_score, gt_labels, cluster_score, pseudo_labels):
        losses = dict()
        #self.weight = 'None'
        print('class_score size:', class_score.size(), 'cluster_score size:', cluster_score.size())
        print('gt_labels size:', gt_labels.size(), 'pseudo_labels size:', pseudo_labels.size())
        losses['loss.{}'.format('_class')] = self.criterion1(class_score, gt_labels)
        losses['acc.{}'.format('_class')] = accuracy(class_score, gt_labels)
        losses['loss.{}'.format('_cluster')] = self.criterion2(cluster_score, pseudo_labels)
        losses['acc.{}'.format('_cluster')] = accuracy(cluster_score, pseudo_labels)
        return losses
