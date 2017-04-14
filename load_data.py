import numpy as np;
import os
import time
import glob
from skimage import exposure,io,transform
import sys
import config as cfg;
import h5py

def get_file_list(dataset='train'):
    """
    for luna16 10 fold data set
    """
    if dataset=='train':
        fs = [];
        ls = [];
        ids = [];
        for s in range(10):
            d = os.path.join(cfg.root,'img_mask','subset{}'.format(s));
            files = glob.glob(os.path.join(d,'image_*.npy'));
            labs = glob.glob(os.path.join(d,'mask_*.npy'));
            fs.extend(files);
            fids = [x.split('/')[-1][6:-4] for x in files];
            lids = [x.split('/')[-1][5:-4] for x in labs];
            ll = [1 if x in lids else 0 for x in fids];
            ls.extend(ll);
            ids.extend([s]*len(ll));

        return fs,ls,ids; #fs=file list, ls:=1 means there is a mask file, or else no (all zero)

class ImgStream:
    def __init__(self, dataset, batch_size, unlabelled_ratio = 1.0):#how many images do we sample from the unlablled images
        self.batch_size = batch_size;
        self.split_id = None
        self.unlabelled_ratio = unlabelled_ratio;
        if dataset == 'train':
            self.imgs,self.labels,self.split_id = get_file_list('train');
            self.labels = np.asarray(self.labels, dtype=np.uint8);
            self.split_id = np.asarray(self.split_id,dtype=np.int);
            # select part of the no label images
            idx = self.labels>0;
            print("labelled image :", np.sum(idx));
            #idx = np.where(np.logical_or(idx,np.random.rand(len(self.labels))<1.0*np.sum(self.labels)/len(self.labels)))[0];
            #self.imgs = [self.imgs[i] for i in idx];
            #self.labels = [self.labels[i] for i in idx];

            print("image data size ", len(self.imgs));
        elif dataset == 'test':
            print("not implemented")
            raise NameError(dataset);
        else:
            raise NameError(dataset);

        self.Tsize = len(self.imgs);
        self.size = int(np.sum(idx) * (1 + self.unlabelled_ratio));

    def __len__(self):
        return self.size;

    def CV_fold_gen(self, fold, NCV, shuffleTrain=True): 
        """
        returns a generator that iterate the expected set
        returns train_gen, val_gen two generators
        """
        idx = np.arange(self.Tsize);
        assert(fold<NCV)
        if self.split_id is not None:
            iii = self.split_id%NCV == fold;
        else:
            iii = idx%NCV == fold;
        train_idx = idx[~iii];
        val_idx = idx[iii];
        return self._datagen(train_idx, cycle=True, shuffle=shuffleTrain), self._datagen(val_idx, cycle=True, shuffle=False);

    def all_gen(self, cycle=True, shuffle=False, test=False):
        idx = np.arange(self.Tsize);
        return self._datagen(idx, cycle, shuffle, test);

    def _datagen(self, input_idx, cycle, shuffle, istest=False):#infinte data generator
        sel_labels = self.labels[input_idx];
        x = np.ones(len(sel_labels));
        x[sel_labels==0] = self.unlabelled_ratio*np.sum(sel_labels>0)/np.sum(sel_labels==0);
        while True:
            # sample from full set of data
            idx = input_idx[np.random.rand(len(x))<x];
            L = len(idx);#dataset size
            Nbatch = L//self.batch_size;

            if shuffle:
                np.random.shuffle(idx)
            for b in range(Nbatch):
                batch_idx = idx[b*self.batch_size:min((b+1)*self.batch_size,L)]
                if cycle and len(batch_idx)<self.batch_size:
                    continue
                yield self._load_data(batch_idx, istest)

            if not cycle:
                break
    def get_mask(self, filename):
        a,b,c = filename.rpartition('image');
        return a+'mask'+c;

    def _load_data(self, batch_idx, test=False):
        X_imgs = [np.load(self.imgs[i]) for i in batch_idx];
        X_imgs = np.array(X_imgs,np.float32)*(1.0/255)-0.05; #to range (0,1) - 0.05 (mean)
        shape = tuple(list(X_imgs.shape)+[1])
        X_imgs = np.reshape(X_imgs,shape);
        if test:
            return X_imgs
        Y_masks = [np.load(self.get_mask(self.imgs[i])) if self.labels[i]>0.5 else np.zeros(X_imgs.shape[1:3],dtype=np.uint8) for i in batch_idx];
        Y_masks = np.array(Y_masks, dtype=np.float32);
        Y_masks = np.reshape(Y_masks,shape)

        return X_imgs,Y_masks

if __name__=='__main__': #test code
    f,l = get_file_list(dataset='train');
    print(len(f),len(l));
    print(f[0:2])
    print(l[0:2])
