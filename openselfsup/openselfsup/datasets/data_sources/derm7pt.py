from ..registry import DATASOURCES
from .image_json_list import ImageJsonList


@DATASOURCES.register_module
class Derm7pt(ImageJsonList):

    def __init__(self, root, list_file, memcached, mclient_path, return_label=True, *args, **kwargs):
        super(Derm7pt, self).__init__(
            root, list_file, memcached, mclient_path, return_label)
