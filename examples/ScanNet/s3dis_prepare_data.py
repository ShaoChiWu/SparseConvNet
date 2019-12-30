############################
# Modify for testing by Wu #
############################

# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob, numpy as np, torch

CLASS_LABELS = {'ceiling':0,
                'floor':1,
                'wall':2,
                'beam':3,
                'column':4,
                'window':5,
                'door':6,
                'table':7,
                'chair':8,
                'sofa':9,
                'bookcase':10,
                'board':11,
                'clutter':12}

files=sorted(glob.glob('/work/carbon537/Stanford3dDataset_v1.2_Aligned_Version/Area_5/*/Annotations/*.txt'))

#print(files)

f = open("/home/carbon537/Area_5_Scene.txt")
scenes = f.read().splitlines()
#print(scenes)
f.close()

for scene in scenes:

    data_list = []
    cat_list = []
    
    for file in files:
        if scene in file:            
            f = open(file)
            datas = f.read().splitlines()
            f.close()
            
            num_items = 0
            
            for data in datas:
                a = [float(x) for x in data.split()]
                data_list.append(a)
                num_items = num_items + 1
                
             
            found = False
            for cat in CLASS_LABELS:
                if cat in file:  
                    found = True
                    for index in range(num_items):
                        cat_list.append(CLASS_LABELS[cat])
                else:
                    pass
            if found == False:
                 for index in range(num_items):
                        cat_list.append(-100)
            else:
                pass
        else:
            pass
        
    v = np.array(data_list).reshape(-1, 6)
    w = np.array(cat_list, dtype=np.float64)
          
    coords=np.ascontiguousarray(v[:,:3]-v[:,:3].mean(0), dtype=np.float32)
    colors=np.ascontiguousarray(v[:,3:6]/127.5-1, dtype=np.float32)
    torch.save((coords,colors,w), '/work/carbon537/S3DIS_data_final/train/Area_5/' + scene + '.pth')
    print(scene)
