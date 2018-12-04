#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:35:55 2018

@author: rishabhbhardwaj
"""

import csv
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CelebA(Dataset):

    def __init__(self, data_dir='./data/celebA/images',
                 annotations_dir='./data/celebA/annotations',
                 split='train', transform=None, target_transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        data_splits = ['train', 'eval', 'test']
        split = data_splits.index(self.split)
        split_data = []
        with open(os.path.join(annotations_dir, 'list_eval_partition.csv')) as split_file:
            reader = csv.reader(split_file, delimiter=',')
            test_row = next(reader)
            for row in reader:
                split_data.append(row)
        bbox_data = []
        with open(os.path.join(annotations_dir, 'list_bbox_celeba.csv')) as bbox_file:
            reader = csv.reader(bbox_file, delimiter=',', skipinitialspace=True)
            test_row = next(reader)  # header row
            #            test_row = next(reader) # header row
            for row in reader:
                bbox_data.append(row)

        split_data = np.array(split_data)
        bbox_data = np.array(bbox_data)
        split_inds = np.where(split_data[:, 1] == str(split))[0]

        self.split_info = split_data[split_inds, :]
        self.bbox_info = bbox_data[split_inds, :]
        self.len = self.split_info.shape[0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        """
        img_name = os.path.join(self.data_dir, self.split_info[index, 0])
        # print(img_name)
        img = Image.open(img_name)
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.len
