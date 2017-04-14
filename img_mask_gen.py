#!/usr/bin/python3
import config as cfg
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
from tqdm import tqdm 
import tools
import os

def mkdir(d):
    try:
        os.makedirs(d)
    except:
        pass

df_node = pd.read_csv(cfg.root+"annotations.csv")

# Getting list of image files
for subset in range(10):
    print("processing subset ",subset)
    luna_subset_path = os.path.join(cfg.root,"data","subset{}".format(subset))
    file_list=sorted(glob(os.path.join(luna_subset_path,"*.mhd")))
    #print(file_list)

    output_path = os.path.join(cfg.root,'img_mask', 'subset{}'.format(subset))
    mkdir(output_path)

    # Looping over the image files in the subset
    for img_file in tqdm(file_list):
        sid = img_file.split('/')[-1][:-4];
        hashid = sid.split('.')[-1];
        sid_node = df_node[df_node["seriesuid"]==sid] #get all nodules associate with file
        #print(sid)
        #print(sid_node)
        
        #load images
        numpyImage, numpyOrigin, numpySpacing = tools.load_itk_image(img_file)
        #load nodules infomation
        #print(numpyImage.shape)
        nodules = [];
        for i in range(sid_node.shape[0]):
            xyz_world = np.array([sid_node.coordX.values[i],sid_node.coordY.values[i],sid_node.coordZ.values[i]]);
            xyz = tools.worldToVoxelCoord(xyz_world, numpyOrigin, numpySpacing);
            d_world = sid_node.diameter_mm.values[i];
            assert numpySpacing[0]==numpySpacing[1]
            d = d_world/numpySpacing[0];
            xyzd = tuple(np.append(xyz,d))
            nodules.append(xyzd)
        h = numpySpacing[2]/numpySpacing[0];
        #print(nodules)

        #Lung mask
        lungMask = tools.segment_lung_mask(numpyImage,speedup=2);
        
        #save images (to save disk, only save every other image/mask pair, and the nodule location slices)
        zs = list(range(1,numpyImage.shape[0],2)) #odd slices
        zs = sorted(zs + [int(x[2]) for x in nodules if x[2]%2==0]);
        minPixels = 0.02*numpyImage.shape[1]*numpyImage.shape[2];
        for z in zs:
            if np.sum(lungMask[z])<minPixels:
                continue
            img,mask = tools.get_img_mask(numpyImage, h, nodules, nth=-1,z=z);
            img = (img*lungMask[z]).astype(np.uint8)
            mask = mask.astype(np.uint8)

            np.save(os.path.join(output_path,"image_%s_%03d.npy" % (hashid, z)),img)
            if np.sum(mask)>1:#if not mask, do not need to save it
                np.save(os.path.join(output_path,"mask_%s_%03d.npy" % (hashid, z)),mask)
        #break
    #break
