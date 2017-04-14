#!/usr/bin/python3
import pickle
import config as cfg
import models
from img_augmentation2 import ImageDataGenerator
import pandas as pd
import numpy as np
from keras import callbacks
from keras import backend as K
import os
import keras.optimizers as opt
from keras.models import load_model
import sys
from tqdm import tqdm

dsb3_root = "/home/qiliu/Share/Clinical/lung/cancer/";

def getmodel(tag, fold):
    model = load_model(os.path.join(cfg.params_dir, 'm3D_{}_{}_fold{}.hdf5'.format(cfg.WIDTH, tag, fold)));
    return model;

if __name__ == '__main__':
    np.random.seed(1234);
    input_tag = '_'.join(sys.argv[2].split(','))
    dd = os.path.join(dsb3_root,'nodule','stage1_3D_{}'.format(input_tag));##!!! input
    dd_out = os.path.join(dsb3_root,'nodule','stage1_3D_{}_sel'.format(input_tag));#output
    if not os.path.exists(dd_out):
        os.mkdir(dd_out);
    cases = os.listdir(dd);

    W = cfg.WIDTH;
    NC = cfg.CHANNEL; #keep the top Channel image
    
    tag = int(sys.argv[1])
    assert(tag==cfg.tag)
    model = getmodel(tag,cfg.fitter['folds'][0]);
    for c in tqdm(cases):
        nod_res,full_img,ss = pickle.load(open('{}/{}'.format(dd,c),'rb'));
        if len(ss)>0:
            X = np.zeros((len(ss),W,W,NC,1),dtype=np.float32);
            for i,img in enumerate(nod_res):
                X[i,:,:,:,0] = img;
            p = [x[0] for x in model.predict(X)];
            ii = np.argsort(p)[::-1];
            iii = ii[0:min(2,len(ii))];#save the most probable two nodules
            res = [nod_res[i] for i in iii]
            imgs = [full_img[i] for i in iii]
            prob = [p[i] for i in iii];
            c0 = [ss[i] for i in iii];
        else:
            res=[];
            imgs = [];
            prob=[];
            c0 = [];

        f = os.path.join(dd_out,c);
        with open(f,'wb') as output:
            pickle.dump([res,imgs,prob,c0], output);
