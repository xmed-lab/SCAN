import os
from PIL import Image

from ..registry import DATASOURCES
from .utils import McLoader
import json

@DATASOURCES.register_module
class ImageJsonList(object):

    def __init__(self, root, list_file, memcached=False, mclient_path=None, return_label=True):
        with open(list_file, 'r') as f:
            self.meta = json.load(f)
        self.has_labels = True
        self.return_label = return_label
        if self.has_labels:
            name_list = []
            label_list = []
            for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
                name_list.append(x)
                label_list.append(y)

            self.fns = name_list
            self.labels = label_list
            self.labels = [int(l) for l in self.labels]
        else:
            # assert self.return_label is False
            self.fns = self.meta['image_names']
        #self.fns = [os.path.join(root, fn) for fn in self.fns]
        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            assert self.mclient_path is not None
            self.mc_loader = McLoader(self.mclient_path)
            self.initialized = True

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        if self.memcached:
            self._init_memcached()
        if self.memcached:
            img = self.mc_loader(self.fns[idx])
        else:
            img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        if self.has_labels and self.return_label:
            target = self.labels[idx]
            return img, target, self.fns[idx]
        else:
            return img, self.fns[idx]
