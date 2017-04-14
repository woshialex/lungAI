from matplotlib import pyplot as plt;
import SimpleITK as sitk
import numpy as np
from skimage.morphology import binary_dilation,square,disk,binary_erosion
from skimage.feature import canny
from skimage import draw,measure,transform
import os

def plot_img_with_mask(img,mask,mask2=None, line_size=2):
    kernel = np.ones((line_size,line_size),dtype=np.uint8)
    if np.max(img)<=1.0:
        img = np.array(img*255,dtype=np.uint8);
    mask = np.array(mask*255, dtype=np.uint8);
    color_img = np.dstack((img,img,img));
    edges = binary_dilation(canny(mask,sigma=1.0),kernel);
    color_img[edges,0] = 255;
    color_img[edges,1] = 0;
    color_img[edges,2] = 0;
    if mask2 is not None:
        mask2 = np.array(mask2*255,dtype=np.uint8);
        edges2 = binary_dilation(canny(mask2,sigma=1.0),kernel);
        color_img[edges2,2] = 255;
        color_img[edges2,0:2] = 0;
    plt.imshow(color_img)

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(itkimage.GetOrigin()) #x,y,z
    numpySpacing = np.array(itkimage.GetSpacing())
     
    return numpyImage, numpyOrigin, numpySpacing

def worldToVoxelCoord(worldCoord, origin, spacing):
    """
    only valid if there is no rotation component
    """     
    voxelCoord = np.rint((worldCoord-origin)/ spacing).astype(np.int);
    return voxelCoord

def normalize(x):
    y = np.copy(x);
    minv = -1000;
    y[x<minv]=minv;
    maxv = 200;
    y[x>maxv]=maxv
    return ((y*1.0-minv)/(maxv-minv)*255).astype(np.uint8);#0-255, to save disk space

def get_img_mask(scan, h, nodules, nth=-1, z=None):
    """
    h = spacing_z/spacing_xy
    nodules = list (x,y,z,r) of the nodule, in Voxel space
    specify nth or z. nth: the nth nodule
    """
    if z is None:
        z = int(nodules[nth][2]);
    img = normalize(scan[z,:,:]);
    res = np.zeros(img.shape);
    #draw nodules
    for xyzd in nodules:
        r = xyzd[3]/2.0;
        dz = np.abs((xyzd[2]-z)*h);
        if dz>=r:continue
        rlayer = np.sqrt(r**2-dz**2);
        if rlayer<3:continue
        #create contour at xyzd[0],xyzd[1] with radius rlayer
        rr,cc=draw.circle(xyzd[1],xyzd[0],rlayer)
        res[rr,cc] = 1;
    return img,res;

def segment_lung_mask2(image, fill_lung_structures=True, speedup=4):    
    def largest_label_volume(im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None
    
    if speedup>1:
        smallImage = transform.downscale_local_mean(image,(1,speedup,speedup));
    else:
        smallImage = image;
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(smallImage > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = np.median([labels[0,0,0],labels[-1,-1,-1],labels[0,-1,-1],
        labels[0,0,-1],labels[0,-1,0],labels[-1,0,0],labels[-1,0,-1]]);

    #Fill the air around the person
    binary_image[background_label == labels] = 2
       
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    m = labels.shape[0]//2;
    l_max = largest_label_volume(labels[m-12:m+20:4,:,:], bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    
    if speedup<=1:
        return binary_image
    else:
        res = np.zeros(image.shape,dtype=np.uint8);
        for i,x in enumerate(binary_image):
            y = transform.resize(x*1.0,image.shape[1:3]);
            res[i][y>0.5]=1
            #res[i] = binary_dilation(res[i],disk(4))
            #res[i] = binary_erosion(res[i],disk(4))
        return res;

def segment_lung_mask(image, speedup=4):
    def largest_label_volume(im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            idx = np.argsort(counts);
            if len(counts)>1:
                return vals[idx[-1]],counts[idx[-2]];
            else:
                return vals[idx[-1]],0
        else:
            return None,None
    if speedup>1:
        smallImage = transform.downscale_local_mean(image,(1,speedup,speedup));
    else:
        smallImage = image;
    c = np.max([-1400,np.percentile(smallImage.flatten(),5)+10]);
    binary_image = np.array((smallImage < -320) & (smallImage>c), dtype=np.int8)
    ##the boundary loop, make it connected
    bbb = np.median(binary_image[:,0:2,:]);
    binary_image[:,0,:]=bbb;
    binary_image[:,-1,:]=bbb;
    binary_image[:,:,0]=bbb;
    binary_image[:,:,-1]=bbb;

    for i, axial_slice in enumerate(binary_image):
        axial_slice = 1-axial_slice
        labeling = measure.label(axial_slice)
        l_max,_ = largest_label_volume(labeling, bg=0)
        if l_max is not None: #This slice contains some lung
            binary_image[i][(labeling!=l_max)] = 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    m = labels.shape[0]//2;
    check_layers = labels[m-12:m+20:4,:,:];
    l_max,c = largest_label_volume(check_layers, bg=0)
        
    while l_max is not None: # There are air pockets
        idx = np.where(check_layers==l_max);
        x,y = idx[1],idx[2];
        #ii = np.vstack(idx[1:]).flatten();
        #if np.max(ii)>labels.shape[1]-24/speedup or np.min(ii)<24/speedup:
        #if np.max([np.std(idx[1]),np.std(idx[2])])>labels.shape[1]/4:
        sz = 30;
        if ((np.min(x)<sz/speedup and (np.min(y)<sz/speedup or np.max(y)>labels.shape[1]-sz/speedup)) or (np.min(x)>labels.shape[1]-sz/speedup and (np.min(y)<sz/speedup or np.max(y)>labels.shape[1]-sz/speedup))) and c>(200/speedup)**2*7:
        #if False:
            binary_image[labels==l_max] = 0;
            labels = measure.label(binary_image, background=0)
            m = labels.shape[0]//2;
            check_layers = labels[m-12:m+20:4,:,:];
            l_max,c = largest_label_volume(check_layers, bg=0)
        else:     
            binary_image[labels != l_max] = 0
            break

    if speedup<=1:
        return binary_image
    else:
        res = np.zeros(image.shape,dtype=np.uint8);
        for i,x in enumerate(binary_image):
            orig = np.copy(x);
            x = binary_dilation(x,disk(5))
            x = binary_erosion(x,disk(5))
            x = np.logical_or(x,orig)            
            y = transform.resize(x*1.0,image.shape[1:3]);
            res[i][y>0.5]=1

        return res;


# Load the scans in given folder path
### DSB3 dataset loading functions
import dicom
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    image[image < -2000] = -2000;
    image[image>2000] = 2000
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float32)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)
