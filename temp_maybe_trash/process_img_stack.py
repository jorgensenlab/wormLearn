## import images

## compute chanels:
    # R: normal
    # G: normal - 00:00
    # B: stdev (variance? abs of range?) of stack

## save

'''
This script has been folded into the track_tools.ImageDataset object
Keeping it around for posterity right now
Or maybe it'll be useful as a way to generate and save the procesed images.

'''

import glob
import os
from PIL import Image
import PIL.ImageOps
import numpy as np
from cv2 import cv2


def img_minus_first(img, first, ch=1):
    '''
    Arguments:
    img: HWC nump array
    first: HWC nump array
    
    Returns:
    a HWC nump array, with img[:,:,ch] - first [:,:,ch]
    '''
    img[:,:,ch] -= first[:,:,ch]
    np.clip(img, 0, 255, img)
    #return clipped

def normalize_stack_intensity(stack, pad_pix=20):
    for i in range(len(stack)-1):
        f1 = stack[i]['i'].astype('float64')
        f2 = stack[i+1]['i'].astype('float64')
        ratio = np.mean(f1[pad_pix:-pad_pix, pad_pix:-pad_pix, :]) / np.mean(f2[pad_pix:-pad_pix, pad_pix:-pad_pix, :])
        stack[i+1]['i'] = f2 * ratio
    stack[0]['i'] = stack[0]['i'].astype('float64')
    return stack


##### ALIGNMENT ON CORNERS
def align_stack(stack, trim_ratio=0.18, pad_pix=10):
    image_h, image_w = stack[0]['i'][:,:,0].shape
    trim_h = round(image_h * trim_ratio)
    trim_w = round(image_w * trim_ratio)
    warp_mode = cv2.MOTION_TRANSLATION
    number_of_iterations = 100
    termination_eps = 1e-7
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                number_of_iterations, termination_eps)
    
    im1 = stack[0]['i']
    TL1 = im1[pad_pix:trim_h, pad_pix:trim_w, 0]
    BL1 = im1[-trim_h:-pad_pix, pad_pix:trim_w, 0]
    TR1 = im1[pad_pix:trim_h, -trim_w:-pad_pix, 0]
    BR1 = im1[-trim_h:-pad_pix, -trim_w:-pad_pix, 0]
    corners1 = [TL1, BL1, TR1, BR1]
    
    for i in range(len(stack)-1):
        im2 = stack[i+1]['i']
        
        TL2 = im2[pad_pix:trim_h, pad_pix:trim_w, 0]
        BL2 = im2[-trim_h:-pad_pix, pad_pix:trim_w, 0]
        TR2 = im2[pad_pix:trim_h, -trim_w:-pad_pix, 0]
        BR2 = im2[-trim_h:-pad_pix, -trim_w:-pad_pix, 0]
        corners2 = [TL2, BL2, TR2, BR2]
        
        warp_matrices = []
        ccs = []
        for c in range(len(corners1)):
            corner1 = corners1[c]
            corner2 = corners2[c]
            
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            # Run the ECC algorithm. The results are stored in warp_matrix.
            (cc, warp_matrix) = cv2.findTransformECC(corner1, corner2, warp_matrix, 
                                                     warp_mode, criteria, None, 1)
            warp_matrices.append(warp_matrix.copy())
            ccs.append(cc)
            
        median_warp = np.median(np.stack(warp_matrices, 2), 2)
        
        target_shape = im1[:,:,0].shape
        
        aligned_image0 = cv2.warpAffine(
                                  im2[:,:,0], 
                                  median_warp, 
                                  (target_shape[1], target_shape[0]), 
                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=0)
        aligned_image1 = cv2.warpAffine(
                                  im2[:,:,1], 
                                  median_warp, 
                                  (target_shape[1], target_shape[0]), 
                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=0)
        aligned_image2 = cv2.warpAffine(
                                  im2[:,:,2], 
                                  median_warp, 
                                  (target_shape[1], target_shape[0]), 
                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=0)
        
        stack[i+1]['i'] = np.stack([aligned_image0, aligned_image1, aligned_image2], axis=2)
    return stack



img_dir = r'F:\Hujber\PyTorch\workspace\wormLearn\images\all\jpeg'
images = sorted(map(os.path.basename, glob.glob(img_dir + '/*.jpeg')))

unique_prefixes = list(set([name.split('_')[0] for name in images]))


img_dic = {}
for prefix in unique_prefixes:
    img_dic[prefix] = {'all': [], '00_00':[]}
    for image in images:
        if prefix in image:
            img = Image.open(os.path.join(img_dir, image))
            inverted_img = PIL.ImageOps.invert(img)
            img_array = np.array(inverted_img)
            img_dic[prefix]['all'].append({'i':img_array, 'name':image})
            if '00_00' in image:
                img_dic[prefix]['00_00'].append({'i':img_array, 'name':image})


for k in img_dic.keys():
    img_dic[k]['all'] = align_stack(img_dic[k]['all'])
    img_dic[k]['all'] = normalize_stack_intensity(img_dic[k]['all'])
    img_dic[k]['00_00'][0]['i'] = np.copy(img_dic[k]['00_00'][0]['i'])
    for im in img_dic[k]['all']:
        img_minus_first(im['i'], img_dic[k]['00_00'][0]['i'])

# Image.fromarray(img_dic['L292R']['all'][0].astype('uint8')).save(os.path.join(img_dir, 'l29r_0ssspos.tif'), compression=None)
# Image.fromarray(img_dic['WZ7nMVEL']['all'][4].astype('uint8')).save(os.path.join(img_dir, 'WZ7nMVEL_1ssspos.tif'), compression=None)

##### SAVE IMGS
for k in img_dic.keys():
    for im in img_dic[k]['all']:
        Image.fromarray(im['i'].astype('uint8')).save(os.path.join(img_dir, 'minus00_aligned', im['name']), quality=100)


