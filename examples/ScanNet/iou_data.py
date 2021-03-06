# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
#scale=20  #Voxel size = 1/scale
scale=50
#val_reps=1 # Number of test views, 1 or more
val_reps=1
#batch_size=32
batch_size = 1
elastic_deformation = False  ## Default: false

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp

dimension=3
full_scale=4096 #Input field size

# Class IDs have been mapped to the range {0,1,...,19}
# NYU_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

val=[]

for x in torch.utils.data.DataLoader(
        glob.glob('/work/carbon537/scannet_raw/val/*.pth'),
        collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
    val.append(x)

print('Validation examples:', len(val))

#Elastic distortion
blur0=np.ones((3,1,1)).astype('float32')/3
blur1=np.ones((1,3,1)).astype('float32')/3
blur2=np.ones((1,1,3)).astype('float32')/3
def elastic(x,gran,mag):
    bb=np.abs(x).max(0).astype(np.int32)//gran+3
    noise=[np.random.randn(bb[0],bb[1],bb[2]).astype('float32') for _ in range(3)]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    ax=[np.linspace(-(b-1)*gran,(b-1)*gran,b) for b in bb]
    interp=[scipy.interpolate.RegularGridInterpolator(ax,n,bounds_error=0,fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x+g(x)*mag

valOffsets=[0]
valLabels=[]
valHeights=[]
for idx,x in enumerate(val):
    #print(x[0][:,2])
    #import sys; sys.exit()
    valOffsets.append(valOffsets[-1]+x[2].size)
    valLabels.append(x[2].astype(np.int32))
    valHeights.append(x[0][:,2].astype(np.float32))
valLabels=np.hstack(valLabels) ##list to array 
valHeights=np.hstack(valHeights)
#print(valHeights[0])
#import sys; sys.exit()

def valMerge(tbl):
    locs=[]
    feats=[]
    labels=[]
    point_ids=[]
    for idx,i in enumerate(tbl):
        a,b,c=val[i]
        m=np.eye(3)
        m[0][0]*=np.random.randint(0,2)*2-1
        m*=scale
        theta=np.random.rand()*2*math.pi
        m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
        a=np.matmul(a,m)+full_scale/2+np.random.uniform(-2,2,3)
        m=a.min(0)
        M=a.max(0)
        q=M-m
        offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
        a+=offset
        idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
        a=a[idxs]
        b=b[idxs]
        c=c[idxs]
        a=torch.from_numpy(a).long()
        locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
        feats.append(torch.from_numpy(b))
        labels.append(torch.from_numpy(c))
        point_ids.append(torch.from_numpy(np.nonzero(idxs)[0]+valOffsets[i]))
    locs=torch.cat(locs,0)
    feats=torch.cat(feats,0)
    labels=torch.cat(labels,0)
    point_ids=torch.cat(point_ids,0)
    return {'x': [locs,feats], 'y': labels.long(), 'id': tbl, 'point_ids': point_ids}
val_data_loader = torch.utils.data.DataLoader(
    list(range(len(val))),batch_size=batch_size, collate_fn=valMerge, num_workers=20,shuffle=False)