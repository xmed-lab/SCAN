from ..registry import DATASOURCES
from .image_json_list import ImageJsonList


@DATASOURCES.register_module
class SD198(ImageJsonList):

    def __init__(self, root, list_file, memcached, mclient_path, return_label=True, *args, **kwargs):
        super(SD198, self).__init__(
            root, list_file, memcached, mclient_path, return_label)
