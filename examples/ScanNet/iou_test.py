# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
#m = 16 # 16 or 32
m=32
#residual_blocks=False #True or False
residual_blocks=True
#block_reps = 1 #Conv block repetition factor: 1 or 2
block_reps=2

import torch, iou
import iou_data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import os, sys, glob
import math
import numpy as np

use_cuda = torch.cuda.is_available()
exp_name='val_iou'

class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(data.dimension,data.full_scale, mode=4)).add(
           scn.SubmanifoldConvolution(data.dimension, 3, m, 3, False)).add(
               scn.UNet(data.dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(data.dimension))
        self.linear = nn.Linear(m, 20)
    def forward(self,x):
        x=self.sparseModel(x)
        x=self.linear(x)
        return x
#print(len(data.val[0]), data.val[0])  

#import sys; sys.exit()
unet=Model()
if use_cuda:
    unet=unet.cuda()
    
pthfile = r'test_model/unet_scale50_m32_rep1_ResidualBlocks_elastic_deformation-000001890-unet.pth'
unet.load_state_dict(torch.load(pthfile))

avgHeights = torch.tensor([0., 0., 0.9983, 0.7615, 0.6757, 0.6576, 0.7369, 1.1313, 1.3664, 1.1504, 1.7234, 0.9957, 0.8069, 1.3108, 0.9460, 1.0056, 0.5083, 0.9097, 0.3493, 0.])

sd = 1.2

while True:
    while True:
        with torch.no_grad():
            unet.eval()
            store=torch.zeros(data.valOffsets[-1],20)
            scn.forward_pass_multiplyAdd_count=0
            scn.forward_pass_hidden_states=0
            start = time.time()
            for rep in range(1,1+data.val_reps):
                for i,batch in enumerate(data.val_data_loader):
                    if use_cuda:
                        batch['x'][1]=batch['x'][1].cuda()
                        batch['y']=batch['y'].cuda()
                    predictions=unet(batch['x'])
                    store.index_add_(0,batch['point_ids'],predictions.cpu())
                    
                    #print(realHeights.size())
                    #print(avgHeights.size())
                    #print(store.size())
                    #dist = store
                    #for i in range(dist.shape[0]):
                    #dist = (realHeights[i]-avgHeights)**2 ## Prior
                    #final = torch.exp(-(dist/(2*sd)))/(np.sqrt(sd)*np.sqrt(2*math.pi)) ## Prior
                    #store = torch.mul(store, final) ## Prior
                realHeights_matrix = np.tile(data.valHeights, (20, 1)).transpose() ## Prior
                realHeights = torch.from_numpy(realHeights_matrix) ## Prior
                dist = (realHeights[:]-avgHeights)**2 ## Prior
                final = torch.exp(-(dist/(2*sd)))/(np.sqrt(sd)*np.sqrt(2*math.pi)) ## Prior
                print(final)
                final[:,0] = 0.3
                final[:,1] = 0.3
                final[:,19] = 0.5
                store = torch.mul(store, final) ## Prior
                iou.evaluate(store.max(1)[1].numpy(),data.valLabels)
        break
    break
