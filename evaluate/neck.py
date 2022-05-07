import torch
import torch.nn as nn
from packaging import version


class NonLinearNeckV0(nn.Module):
    """The non-linear neck in ODC, fc-bn-relu-dropout-fc-relu.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 sync_bn=False,
                 with_avg_pool=True):
        super(NonLinearNeckV0, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.fc0 = nn.Linear(in_channels, hid_channels)
        
        self.bn0 = nn.BatchNorm1d(
            hid_channels, momentum=0.001, affine=False)

        self.fc1 = nn.Linear(hid_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.sync_bn = sync_bn

    #def init_weights(self, init_linear='normal'):
        #_init_weights(self, init_linear)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        print("====x size====:", len(x))
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn0, x)
        else:
            x = self.bn0(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        return [x]