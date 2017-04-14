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
import sys

def train(Xtrain,Ytrain,Xval,Yval,fold):
    #call backs
    tag = cfg.tag;
    model_checkpoint = callbacks.ModelCheckpoint(
            os.path.join(cfg.params_dir, 'm3D_{}_{}_fold{}.hdf5'.format(cfg.WIDTH, tag, fold)), 
            monitor='val_loss', verbose=0,
            save_best_only=False
            );
    learn_rate_decay = callbacks.ReduceLROnPlateau(
            monitor='val_loss',factor=0.3, 
            patience=8, min_lr=1e-5, verbose=1
            );
    earlystop = callbacks.EarlyStopping(monitor='val_loss',patience=15,mode='min',verbose=1);

    ## build the neural net work
    model =  models.get_3Dnet();
    model.compile(
            optimizer= eval("opt."+cfg.fitter['opt'])(**cfg.fitter['opt_arg']),
            loss='binary_crossentropy'
            );
    datagen = ImageDataGenerator(**cfg.aug);
    datagenOne = ImageDataGenerator();

    #Fit here
    history = model.fit_generator(datagen.flow(Xtrain,Ytrain,batch_size=cfg.fitter['batch_size']), 
            steps_per_epoch=len(Xtrain)//cfg.fitter['batch_size'],
            epochs=cfg.fitter['num_epochs'],
            validation_data = datagenOne.flow(Xval,Yval,batch_size=cfg.fitter['batch_size']),
            validation_steps = len(Xval)//cfg.fitter['batch_size'],
            callbacks = [learn_rate_decay, model_checkpoint, earlystop]
            );
    del model


if __name__ == '__main__':
    np.random.seed(123);
    tag = sys.argv[1].split(',')
    nods_dir = '/home/qiliu/Share/Clinical/lung/luna16/nodule_candidate_{}/'.format('_'.join(tag));
    ids = os.listdir(nods_dir);
    data = [pickle.load(open('%s/%s'%(nods_dir, i),'rb')) for i in ids];
    Ncase = np.sum([len(x[0]) for x in data]);
    
    Y = np.zeros(Ncase);
    W = cfg.WIDTH;
    NC = cfg.CHANNEL; 
    X = np.zeros((len(Y),W,W,NC),dtype=np.float32); #set to 0 for empty chanels
    c = 0;
    for x in data:
        nx = len(x[0]);
        if nx==0:
            continue
        imgs = x[0];
        ls = x[1];
        Y[c:c+nx] = ls;
        X[c:c+nx] = imgs;
        c += nx;
    
    print("total training cases ", Ncase);
    print("percent Nodules: ",np.sum(Y)/Ncase);

    # N fold cross validation
    NF = cfg.fitter['NCV'];
    ii =  np.arange(Ncase);
    for i in cfg.fitter['folds']:
        ival = (ii%NF==i);
        Xtrain= X[~ival];
        Ytrain = Y[~ival];
        Xval = X[ival];
        Yval = Y[ival];
        train(Xtrain,Ytrain,Xval,Yval,i);
