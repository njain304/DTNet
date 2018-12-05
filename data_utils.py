#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:40:30 2018

@author: rishabhbhardwaj
"""
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np


class ResizeTransform(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample.resize((self.size, self.size), Image.BILINEAR)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img)


class ZeroPadding(object):
    def __init__(self, size, use_gpu=True):
        self.size = size
        self.use_gpu = use_gpu

    def __call__(self, sample):
        B, C, H, W = sample.size()
        diff = self.size - H
        padding = Variable(torch.zeros(B, C, diff, W), requires_grad=False)
        if self.use_gpu:
            padding = padding.cuda()
        zero_padded = torch.cat((sample, padding), dim=2)
        return zero_padded


class TanhNormalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample = sample * 2.0 - 1.0
        return sample


class TanhDeNormalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample = (sample + 1.0) * 0.5
        return sample


class UnNormalize(object):
    def __init__(self, mean, std):
        mean_arr = []
        for dim in range(len(mean)):
            mean_arr.append(dim)
        std_arr = []
        for dim in range(len(std)):
            std_arr.append(dim)
        self.mean = torch.Tensor(mean_arr).view(1, len(mean), 1, 1)
        self.std = torch.Tensor(std_arr).view(1, len(std), 1, 1)

    def __call__(self, tensor):
        tensor *= self.std
        tensor += self.mean
        return tensor
