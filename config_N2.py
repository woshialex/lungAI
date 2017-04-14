import sys
import os
tag = 2
#not working better
net_version = 'Simple3D_2'
net_params = None

#score
## -------------------------------------------------
## ---- data files ----
## -------------------------------------------------
root="/home/qiliu/Share/Clinical/lung/luna16/"

#make these directories
cache_dir = os.path.join(root, 'cache')
params_dir = os.path.join(root, 'params')
res_dir = os.path.join(root, 'results')
log_dir = os.path.join(root, 'log');
for d in [cache_dir,params_dir, res_dir, log_dir]:
    if not os.path.isdir(d):
        os.makedirs(d)

## -------------------------------------------------
## -----  fitter -------
## -------------------------------------------------
#sample
fitter = dict(
    batch_size = 16,
    num_epochs = 100,
    NCV = 5,
    folds = [1],
    opt = 'Adam',
    opt_arg =dict(lr=0.2e-3),
);

## -------------------------------------------------
## ----- model specification ----
## -------------------------------------------------
#image size
WIDTH=64;
HEIGHT=64;
CHANNEL=16;

## -------------------------------------------------
## ----- image augmentation parameters ----
## -------------------------------------------------
aug = dict(
	 featurewise_center=False,
	 samplewise_center=False,
	 featurewise_std_normalization=False,
	 samplewise_std_normalization=False,
	 zca_whitening=False,
	 rotation_range=15.0,
	 width_shift_range=0.10,
	 height_shift_range=0.10,
	 shear_range=0.05,
	 zoom_range=0.10,
	 channel_shift_range=0.,
	 fill_mode='constant',
	 cval=0.,
	 horizontal_flip=True,
	 vertical_flip=False,
	 rescale=None,
);
