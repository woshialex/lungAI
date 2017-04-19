#!/usr/bin/python3
import pickle
import config as cfg
import models
from img_augmentation import ImageDataGenerator
import pandas as pd
import numpy as np
from keras import callbacks
from keras import backend as K
import os
import keras.optimizers as opt
import lsuv_init

def getmodel(tag, fold):
    ## for initialiation
    from keras.models import load_model
    model = load_model(os.path.join('/home/qiliu/Share/Clinical/lung/luna16/params', 'm3D_{}_{}_fold{}.hdf5'.format(cfg.WIDTH, tag, fold)));
    return model;

def train(Xtrain,Ytrain,Xval,Yval,fold):
    #call backs
    tag = cfg.tag;
    model_checkpoint = callbacks.ModelCheckpoint(
            os.path.join(cfg.params_dir, 'm3D_{}_{}_fold{}.hdf5'.format(cfg.WIDTH, tag, fold)), 
            monitor='val_loss', verbose=0,
            save_best_only=True
            );
    learn_rate_decay = callbacks.ReduceLROnPlateau(
            monitor='val_loss',factor=0.3, 
            patience=10, min_lr=1e-6, verbose=1
            );
    earlystop = callbacks.EarlyStopping(monitor='val_loss',patience=25,mode='min',verbose=1);

    ## build the neural net work
    #model =  models.get_3Dnet();

    #or load from pretrained
    model = getmodel(1,0);
    
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
            validation_steps= len(Xval)//cfg.fitter['batch_size'],
            callbacks = [learn_rate_decay, model_checkpoint, earlystop]
            );
    del model


if __name__ == '__main__':
    np.random.seed(123);
    labels = pd.read_csv('/home/qiliu/Share/Clinical/lung/cancer/train/stage1_labels.csv');
    data = [pickle.load(open('/home/qiliu/Share/Clinical/lung/cancer/nodule/stage1_3D_%s/%s.pkl'%(cfg.data_version,labels.id[i]),'rb')) for i in range(labels.shape[0])];
    labels['counts'] = [len(x[1]) for x in data];
    
    idx = labels['counts'].values>0;
    ##
    noNodule = labels[~idx];
    p = np.sum(noNodule.cancer>0.5)*1.0/noNodule.shape[0];
    print("noNodule percentage ", noNodule.shape[0]*1.0/labels.shape[0]);
    print("noNodule cancer probability ",p);
    ##train data, those has nodule detections, for the rest, all predict the same value
    Y = labels.cancer[idx].values;
    pp = np.sum(Y>0.5)/len(Y);
    print("cancer probability ",pp);
    W = cfg.WIDTH;
    NC = cfg.CHANNEL; 
    X = np.zeros((len(Y),W,W,NC),dtype=np.float32); #set to 0 for empty chanels
    c = 0;
    for i in np.where(idx)[0]:
        imgs = data[i][0];
        #!!! only take most probable nodule....
        for img in imgs:
            if img.shape!=(cfg.WIDTH,cfg.HEIGHT,cfg.CHANNEL):
                print(i,"shape ==",img.shape);
                continue
            else:
                X[c] = img;
                break
        c += 1
    #rm mean?
    #Xm = np.mean(X);
    #print('X mean: ', Xm);
    #X = X - Xm;
    
    NCase = len(Y);
    print("total training cases ", NCase);
    # N fold cross validation
    NF = cfg.fitter['NCV'];
    ii =  np.arange(NCase);
    for i in cfg.fitter['folds']:
        ival = (ii%NF==i);
        Xtrain= X[~ival];
        Ytrain = Y[~ival];
        Xval = X[ival];
        Yval = Y[ival];
        train(Xtrain,Ytrain,Xval,Yval,i);
