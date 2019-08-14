import os, os.path,shutil
import sys

with open('training_data.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line=line.strip('\n')
        #print(line)
        oldpath = 'scans/'
        path_label = oldpath + line + '/' + line + '_vh_clean_2.labels.ply'
        path = oldpath + line + '/' + line + '_vh_clean_2.ply'
        #print(path)
        newpath = 'train/'
        if os.path.exists(path) and os.path.exists(path_label):
            shutil.move(path, newpath)
            shutil.move(path_label, newpath)