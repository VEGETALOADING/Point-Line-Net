from typing import List, Tuple
from torch import Tensor


class ImageList(object):


    def __init__(self, tensors, image_sizes):

        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

