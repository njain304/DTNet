#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:56:21 2018

@author: rishabhbhardwaj
"""

from datasets.simpsons import Simpsons
from datasets.celebA import CelebA
import data_utils

import torch
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import numpy as np

simp_transform = transforms.Compose([
            data_utils.ResizeTransform(96), 
            data_utils.NormalizeRangeTanh()
            #transforms.ToPILImage(),
            #transforms.ToTensor()
        ])

simp_train_set = Simpsons(transform = simp_transform)

simp_loader = torch.utils.data.DataLoader(simp_train_set, batch_size=1, shuffle=True, num_workers=8)

simp_iter = iter(simp_loader)
simp_img = simp_iter.next()
print(simp_img.size())
plt.imshow(np.transpose(simp_img.numpy()[0], (1, 2, 0)))


celebA_train_set = CelebA(transform = simp_transform)
celebA_loader = torch.utils.data.DataLoader(celebA_train_set, batch_size=1, shuffle=True, num_workers=8)
celebA_iter = iter(celebA_loader)
celebA_img = celebA_iter.next()
print(celebA_img.size())
plt.imshow(np.transpose(celebA_img.numpy()[0], (1, 2, 0)))
