#!/usr/bin/python3
import pandas as pd
import sys
import config as cfg;
from keras.models import load_model
import utils
from glob import glob
import tools
import nibabel as nib
import os
from tqdm import tqdm 
import numpy as np
from skimage import measure
import utils
import pickle

### loss fuction to make getmodel work, copied from segment/utils.py
def getmodel(tag, fold):
    model = load_model(os.path.join(cfg.params_dir, 'unet_{}_{}_fold{}.hdf5'.format(512, tag, fold)), \
                       custom_objects={'loss':utils.dice_coef_loss_gen(smooth=10)});
    return model;

## output nodule image size
W = 64;
H = 16;

df_node = pd.read_csv(cfg.root+"annotations.csv")

assert(len(sys.argv)==2);
tags = sys.argv[1].split(',');
models = [getmodel(int(t),0) for t in tags];

# Getting list of image files
output_path = os.path.join(cfg.root,'nodule_candidate_%s'%('_'.join(tags)));
if not os.path.exists(output_path):
    os.mkdir(output_path)
case = 0;
for subset in range(10):
    print("processing subset ",subset)
    luna_subset_path = os.path.join(cfg.root,"data","subset{}".format(subset))
    file_list=sorted(glob(os.path.join(luna_subset_path,"*.mhd")))
    #print(file_list)

    # Looping over the image files in the subset
    #for img_file in tqdm(file_list):
    for img_file in file_list:
        case += 1
        sid = img_file.split('/')[-1][:-4];
        hashid = sid.split('.')[-1];
        sid_node = df_node[df_node["seriesuid"]==sid] #get all nodules associate with file
        
        #load images
        numpyImage, numpyOrigin, numpySpacing = tools.load_itk_image(img_file)
        #load nodules infomation
        nd_list = [];
        for i in range(sid_node.shape[0]):
            xyz_world = np.array([sid_node.coordX.values[i],sid_node.coordY.values[i],sid_node.coordZ.values[i]]);
            xyz = tools.worldToVoxelCoord(xyz_world, numpyOrigin, numpySpacing);
            d_world = sid_node.diameter_mm.values[i];
            assert numpySpacing[0]==numpySpacing[1]
            d = d_world/numpySpacing[0];
            xyzd = tuple(np.append(xyz,d))
            nd_list.append(xyzd)
        h = numpySpacing[2]/numpySpacing[0];
        #print(nd_list)

        imgs = numpyImage;
        lungMask = tools.segment_lung_mask(imgs,speedup=2);
        res = np.array(tools.normalize(imgs)*lungMask,np.float32)*(1.0/255)-0.05;
        shape = tuple(list(res.shape)+[1]);
        res = np.reshape(res,shape);
        cut = cfg.keep_prob;
        nodules = models[0].predict(res,batch_size=8)>cut;
        for model in models[1:]:
            nodules += model.predict(res,batch_size=8)>cut;
        shape = nodules.shape[:-1];
        nodules = np.reshape(nodules,shape)*(lungMask>0.2);

        ## find nodules and it's central location
        binary_nodule = np.array(nodules >=0.5, dtype=np.int8);
        labels = measure.label(binary_nodule)
        vals, counts = np.unique(labels, return_counts=True)
        counts = counts[vals!=0]
        vals = vals[vals!=0]
        nl = sorted(zip(counts,vals),reverse=True);

        ## larger than 100 pixels in volume
        volume_cut = 50;
        nl = [x for x in nl if x[0]>volume_cut];
        
        nod_res =[];
        ls = [];
        Ndeteced = 0;
        for nod in range(len(nl)):
            xyz = np.where(labels==nl[nod][1]);
            c = tuple(int(np.median(x)) for x in xyz);
            ss = tuple(int(np.max(x)-np.min(x))+1 for x in xyz);
            #if ss[0]<3: #thickness<3 pixels
            #    continue
            # output result image with WIDTH
            if c[1]-W//2<=0 or c[1]+W//2>512 or c[2]-W//2<=0 or c[2]+W//2>512 or c[0]-H//2<=0 or c[0]+H//2>=res.shape[0]:
                continue
            try:
                out = np.transpose(res[c[0]-H//2:c[0]+H//2,c[1]-W//2:c[1]+W//2,c[2]-W//2:c[2]+W//2,0]+0.05, (1,2,0));
            except:
                continue
            nod_res.append(out);
            l = 0;
            for xyzd in nd_list:
                dz = np.abs(xyzd[2]-c[0])*h;
                dx = np.abs(xyzd[0]-c[2]);
                dy = np.abs(xyzd[1]-c[1]);
                if dx*dx+dy*dy+dz*dz <= xyzd[3]**2/4:
                    l=1;
                    Ndeteced += 1
                    break
            ls.append(l);
        print(case,len(nd_list),len(ls),Ndeteced)

        # save 
        f = os.path.join(output_path,'{}.pkl'.format(hashid));
        with open(f,'wb') as output:
            pickle.dump([nod_res,ls], output);
