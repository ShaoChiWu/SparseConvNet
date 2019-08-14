import os, os.path,shutil
import sys

with open('val_list.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line=line.strip('\n')
        print(line)
        oldpath = 'val/'
        path = oldpath + line + '_vh_clean_2.ply'
        #print(path)
        newpath = 'val100/'
        if os.path.exists(path):
            shutil.move(path, newpath)
            
