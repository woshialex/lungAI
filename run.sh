#PID=32588
#while s=`ps -p $PID -o s=` && [[ "$s" && "$s" != 'Z' ]]; do
#	sleep 1
#done
##pipeline

#preprocess
./img_mask_gen.py #geneate image(lung)/mask(nodule) pairs 

##train (use load_data.ImgStream to load image and feed to ImageAugment to do augmentation)
rm config.py
ln -s config_v11.py config.py
./train.py
./NoduleDetect.py 11

##train a 3D model to classifier a region is a nodule or not.
rm config.py
ln -s config_N2.py config.py
./trainNodule.py 11
###use the trained model, to produce a probability on the possible nodules deteced by ./DSB3NoduleDetect.py, and then build a cancer detection model
./predNodule.py 2 11
