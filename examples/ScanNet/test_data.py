############################
# Modify for testing by Wu #
############################

# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
#scale=20  #Voxel size = 1/scale
scale = 20
#val_reps=1 # Number of test views, 1 or more
test_reps = 1
#batch_size=32
batch_size = 1
elastic_deformation = False

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp

dimension = 3
full_scale = 4096 #Input field size

# Class IDs have been mapped to the range {0,1,...,19}
# NYU_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

test=[]

test_list = []
test_list = sorted(glob.glob('/work/carbon537/S3DIS_data_final/val/Area_6_no_label/*.pth'))

for x in torch.utils.data.DataLoader(
        test_list,
        collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
    #print(len(x[0]))
    test.append(x)
    
print('Testing examples:', len(test))

#import sys;sys.exit(0);

MyFile=open('area_6_test_s3dis_scene_order.txt','w')
test_list=map(lambda x:x+'\n', test_list)
MyFile.writelines(test_list)
MyFile.close()

# import sys; sys.exit()

testOffsets=[0]
for idx,x in enumerate(test):
    testOffsets.append(testOffsets[-1]+len(x[1]))
#    print(testOffsets[-1]+x[1].size)
#     if idx == 3:
#         import sys;sys.exit(0);
#print(testOffsets, len(testOffsets))
#import sys;sys.exit(0);

def testMerge(tbl):
    locs=[]
    feats=[]
    point_ids=[]
    for idx,i in enumerate(tbl):
        a,b=test[i]
        #print(np.shape(a))
        #import sys;sys.exit(0);
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
        #print(np.shape(a))
        a=a[idxs]
        b=b[idxs]
        #print(np.shape(a))
        #import sys;sys.exit(0);
        a=torch.from_numpy(a).long()
        locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
        feats.append(torch.from_numpy(b))
        point_ids.append(torch.from_numpy(np.nonzero(idxs)[0]+testOffsets[i]))
    locs=torch.cat(locs,0)
    feats=torch.cat(feats,0)
    point_ids=torch.cat(point_ids,0)
#     print(locs.shape, locs[0], locs[1], locs[-1])
#     print(feats.shape, feats[0], feats[1], feats[-1])
#     print(point_ids.shape, point_ids[0], point_ids[1], point_ids[-1])
#     import sys;sys.exit(0);
    return {'x': [locs,feats], 'id': tbl, 'point_ids': point_ids}
test_data_loader = torch.utils.data.DataLoader(
    list(range(len(test))),batch_size=batch_size, collate_fn=testMerge, num_workers=20,shuffle=False)
