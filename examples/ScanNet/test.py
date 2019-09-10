############################
# Modify for testing by Wu #
############################

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

import torch
#import test_iou as iou
import test_data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import os, sys, glob
import math
import numpy as np
import torchvision.models as models
import scipy.io as io

use_cuda = torch.cuda.is_available()
exp_name='val311_unet_scale50_m32_rep1_ResidualBlocks'

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

unet=Model()
if use_cuda:
    unet=unet.cuda()
    
pthfile = r'test_model/unet_scale50_m32_rep1_ResidualBlocks_elastic_deformation-000000512-unet.pth'
unet.load_state_dict(torch.load(pthfile))

with open("val311.txt",'r') as fp:
    all_lines = fp.readlines()

all_lines.sort()
print(all_lines)
    
with torch.no_grad():
    unet.eval()
    store=torch.zeros(data.testOffsets[-1],20)
    #print(store, store.shape)
#     for i in range(100):
#         print(data.testOffsets[i], data.testOffsets[100])
    #import sys; sys.exit(0);
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    start = time.time()
    for rep in range(1,1+data.test_reps):
        for i,batch in enumerate(data.test_data_loader):
            if use_cuda:
                batch['x'][1]=batch['x'][1].cuda()
                #print(len(batch['x'][1]))
                #import sys; sys.exit(0);
            predictions=unet(batch['x'])
            #store.index_add_(0,batch['point_ids'],predictions.cpu())
            ttt = predictions.cpu()
            result = torch.max(ttt, 1)[1]
            np.set_printoptions(formatter={'all':lambda x: str(x)}) #avoid sci. notation 
            result_a = np.array(result)
            np.savetxt(all_lines[i][:-1]+'.txt',result_a, fmt='%d')
            
            #print(ttt.shape, ttt[0][0].dtype)
            #import sys; sys.exit(0);
#             print(batch['point_ids'].max(), batch['point_ids'].min(), batch['point_ids'].shape)
#             if i==2:
#                 import sys; sys.exit(0);
            #np.savetxt(str(i+707),store.max(1)[1].numpy())
        print('time=',time.time() - start,'s')
        #iou.evaluate(store.max(1)[1].numpy(),data.testLabels)
        #print(type(store.max(1)[1]))
        #print(type(store.max(1)[1].numpy()))
        #np.savetxt('result',store.max(1)[1].numpy())