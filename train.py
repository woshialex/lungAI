#!/usr/bin/python3
import os
import Unet
import numpy as np
import pandas as pd
import config as cfg
import load_data as ld
from img_augmentation import ImageAugment
import time
import sys
from keras import callbacks
from keras import backend as K
import keras.optimizers as opt
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
import utils

if __name__=='__main__':
    """
    build and train the CNNs.
    """
    np.random.seed(1234);
    tag = cfg.tag;

    #load data, flow
    full_train_data = ld.ImgStream("train", cfg.fitter['batch_size'], unlabelled_ratio = cfg.unlabelled_ratio);

    #image aumentation, flow
    imgAug = ImageAugment(**cfg.aug);

    folds_best_epoch = [];
    folds_history = {};
    for fold in cfg.fitter['folds']:
        assert(fold<=cfg.fitter['NCV']);
        print("--- CV for fold: {}".format(fold))
        if fold<cfg.fitter['NCV']:#CV
            train_data, val_data  = full_train_data.CV_fold_gen(
                    fold, cfg.fitter['NCV'], shuffleTrain=True);
            num_epochs = cfg.fitter['num_epochs'];
        else:#all training data
            train_data = full_train_data.all_gen(shuffle=True);
            num_epochs = int(np.mean(folds_best_epoch)+1) if len(folds_best_epoch)>0 else cfg.fitter['num_epochs'];
            val_data = None;

        train_data_aug = imgAug.flow_gen(train_data, mode='fullXY');

	#call backs
        model_checkpoint = callbacks.ModelCheckpoint(
                os.path.join(cfg.params_dir, 'unet_{}_{}_fold{}.hdf5'.format(cfg.WIDTH, tag, fold)), 
                monitor='val_loss', verbose=0,
                save_best_only=False
                );
        learn_rate_decay = callbacks.ReduceLROnPlateau(
                monitor='val_loss',factor=0.3, 
                patience=8, min_lr=1e-6, verbose=1
                );
        earlystop = callbacks.EarlyStopping(monitor='val_loss',patience=15,mode='min',verbose=1);
        tensorboard = callbacks.TensorBoard(log_dir=cfg.log_dir,histogram_freq=10,write_graph=False);

        ## build the neural net work
        model =  Unet.get_unet(cfg.HEIGHT, cfg.WIDTH, cfg.net_version);
        model.compile(
                optimizer= eval("opt."+cfg.fitter['opt'])(**cfg.fitter['opt_arg']),
                loss=utils.dice_coef_loss_gen(smooth=cfg.smooth, pred_mul=cfg.pred_loss_mul, p_ave=cfg.p_ave, p_cross=cfg.p_cross)
                );

	#Fit here
        history = model.fit_generator(train_data_aug, 
                steps_per_epoch=(3000//cfg.fitter['batch_size']),
                epochs =cfg.fitter['num_epochs'],
                validation_data = val_data, 
                validation_steps = len(full_train_data)//cfg.fitter['NCV']//cfg.fitter['batch_size'],
                callbacks = [learn_rate_decay, model_checkpoint, earlystop]
                );
        #history has epoch and history atributes
        folds_history[fold] = history.history;
        folds_best_epoch.append(np.argmin(history.history['val_loss']));
        del model

    #save the full fitting history file for later study
    import pickle
    with open(os.path.join(cfg.log_dir,'history_{}_{}.pkl'.format(cfg.WIDTH, tag)),'wb') as f:
        pickle.dump(folds_history, f);
    #summary of all folds
    summary = utils.hist_summary(folds_history);
    print(summary)
