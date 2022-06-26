import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class HWSegmentation(data.Dataset):
    """
    Load HandWriting Segmention
    """

    HandWClass = namedtuple('HandWClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                           'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        HandWClass('background', 0, 0, 'void', 0, False, False, (255, 255, 255)),
        HandWClass('hand', 1, 1, 'void', 0, False, False, (128, 64, 128)),
        HandWClass('print', 2, 2, 'void', 0, False, False, (244, 35, 232)),
    ]

    train_id_to_color = [c.color for c in classes]
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, root, transform=None, train=True):
        self.images = os.listdir(os.path.join(root, 'Images'))
        self.targets = os.listdir(os.path.join(root, 'Labels'))
        print(root, " images number:", len(self.images))
        for item in self.images:
            if item not in self.images:
                self.images.remove(item)

        self.images.sort()
        self.targets.sort()
        self.images = [os.path.join(root, 'Images', item) for item in self.images]
        self.targets = [os.path.join(root, 'Labels', item) for item in self.targets]
        if train:
            self.images = self.images[0: int(len(self.images) * 0.85)]
            self.targets = self.targets[0: int(len(self.targets) * 0.85)]
        else:
            self.images = self.images[int(len(self.images) * 0.85):]
            self.targets = self.targets[int(len(self.targets) * 0.85):]
        self.transform = transform

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform is not None:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)
