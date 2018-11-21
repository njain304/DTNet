#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:56:21 2018

@author: rishabhbhardwaj
"""

from datasets.simpsons import Simpsons
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
plt.show()