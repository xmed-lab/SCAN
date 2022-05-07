from PIL import Image
from .registry import DATASETS
from .base import BaseDataset
import torch
from openselfsup.utils import print_log

@DATASETS.register_module
class SCANDataset(BaseDataset):
    """Dataset for SCAN"""

    def __init__(self, data_source, pipeline):
        super(SCANDataset, self).__init__(data_source, pipeline)
        # init clustering labels, psuedo-label
        self.labels = [-1 for _ in range(self.data_source.get_length())]

    def __getitem__(self, idx):
        img, target, filename = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        p_label = self.labels[idx]
        img = self.pipeline(img)
        return dict(img=img, pseudo_label=p_label, gt_label=target, idx=idx)

    def assign_labels(self, labels):
        assert len(self.labels) == len(labels), \
            "Inconsistent length of assigned labels, \
            {} vs {}".format(len(self.labels), len(labels))
        self.labels = labels[:]

    #def evaluate(self, scores, keyword, logger=None):

    def evaluate(self, scores, keyword, logger=None, topk=(1, 5)):
        eval_res = {}

        target = torch.LongTensor(self.data_source.labels)
        assert scores.size(0) == target.size(0), \
            "Inconsistent length for results and labels, {} vs {}".format(
            scores.size(0), target.size(0))
            
        num = scores.size(0)
        _, pred = scores.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # KxN
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0).item()
            acc = correct_k * 100.0 / num
            eval_res["{}_top{}".format(keyword, k)] = acc
            if logger is not None and logger != 'silent':
                print_log(
                    "{}_top{}: {:.03f}".format(keyword, k, acc),
                    logger=logger)
        return eval_res
