import trakerManager
from PIL import Image
from PIL.ImageOps import invert, autocontrast
import matplotlib.pyplot as plt
import visualization_utils as vis_utils
import numpy as np
import os
import utils
import pandas as pd
import glob
from datetime import datetime
import pascal_voc_io
from lableFile import LabelFile

class PascalVocReader(pascal_voc_io.PascalVocReader):
    def getInference(self):
        return dict(boxes=self.getBoxes(),
                    labels=self.getLabels(),
                    scores=self.getScores())
    
    def getBoxes(self):
        return np.array([self.points_to_bbox(shape) for shape in self.shapes])
    
    def getLabels(self):
        return np.ones(len(self.shapes), dtype='uint64')
        
    def getScores(self):
        return np.array([shape[0] for shape in self.shapes], dtype='float32')
    
    def points_to_bbox(self, shape):
        x1 = min(p[0] for p in shape[1])
        y1 = min(p[1] for p in shape[1])
        x2 = max(p[0] for p in shape[1])
        y2 = max(p[1] for p in shape[1])
        return np.array([x1, y1, x2, y2], dtype='float32')


def saveLabels(imgFilePath, inference, imageShape=(1952,2592,1)):
    annotationFilePath = os.path.splitext(imgFilePath)[0] + '.xml'
    labelFile = LabelFile()
    
    def format_shape(s):
        return dict(label=s[1],
                    #line_color=s.line_color.getRgb(),
                    #fill_color=s.fill_color.getRgb(),
                    points=bbox_to_points(s[0]),
                    # add chris
                    difficult = 0)
    
    def bbox_to_points(b):
        x1, y1, x2, y2 = b       
        return [(x1,y1), (x1,y2), (x2,y1), (x2,y2)]
        
    bboxes = zip(inference['boxes'], inference['scores']) 
    shapes = [format_shape(shape) for shape in bboxes]
    labelFile.savePascalVocFormat(annotationFilePath, shapes, imgFilePath, imageShape)

        

def loadLabels(imgFilePath):
    annotationFilePath = os.path.splitext(imgFilePath)[0] + '.xml'
    tVocParseReader = PascalVocReader(annotationFilePath)
    shapes = tVocParseReader.getInference()





start = datetime.now()

##################################
PATH_TO_MODEL = 'F:\\Hujber\\PyTorch\\workspace\\wormLearn\\runs\\19-10-17_19-28_bellahidden_reduceLRonplateau\\trained_model.pth'
PATH_TO_DATA_FOLDERS =  r'E:\\aldicarb_images\\2019_08_29'
##################################

data_folders = glob.glob(PATH_TO_DATA_FOLDERS+"/*/")


''' 
Normal images come in "worm dark" so load_and_process_stack always inverts it
because "Worm light" is necessary for img subraction and proper inference. Additional inversion by the 
transforms would be inappropriate. 
'''
file_type = 'tif'  
datasets = [track_tools.ImageDataset(path, track_tools.inference_transforms(), file_type=file_type) for path in data_folders]

inference_results = track_tools.infer_worm_BBoxes(PATH_TO_MODEL, datasets)
inf_time =datetime.now()


## Save bounding boxes in pascal VOC format. This is compatible with lableImg
for dataset in datasets:
    results = inference_results[dataset.root]
    for img_path, inference in results.items():
        saveLabels(img_path, inference)


#####
#####   Maybe you edit some bounding boxes now?
#####

## Load in edited bounding boxes (pascal VOC. xml output form labelImg)
for dataset in datasets:
    results = inference_results[dataset.root]
    for img_path, inference in results.items():
        inference = loadLabels(img_path)




## Find center of mass of each worm
##### Images are stored as "worm light," and thats what we ened for proper thresholding here. So no inversion! 
need_invert = False

for dataset in datasets:
    results = inference_results[dataset.root]
    for img_path, inference in results.items():
        img = dataset.get_by_name_as_gray_Image(img_path)  ## just the first channel
        inference['CoM'] = []
        inference['Image'] = img
        for i in range(inference['boxes'].shape[0]):
            left   = int(inference['boxes'][i, 0])
            top    = int(inference['boxes'][i, 1])
            right  = int(inference['boxes'][i, 2])
            bottom = int(inference['boxes'][i, 3])
            score  = inference['scores'][i]
            
            cropped = img.crop((left, top, right, bottom))
            center_of_mass, center_of_pix = track_tools.get_center_of_worm(cropped)
            center_of_mass_global = (center_of_mass[1] + left , center_of_mass[0] + top)
            inference['CoM'].append(center_of_mass_global)


## Calculate px distance to nearest CoM in next image
for dataset in datasets:
    results = inference_results[dataset.root]
    sorted_imgs = sorted(list(results.keys()))
    results[sorted_imgs[0]]['min_dist_px'] = []
    for i in range(len(sorted_imgs)-1):
        CoMs1 = results[sorted_imgs[i]]['CoM']
        CoMs2 = results[sorted_imgs[i+1]]['CoM']
        results[sorted_imgs[i+1]]['min_dist_px'] = []
        for j in range(len(CoMs1)):
            results[sorted_imgs[i]]['min_dist_px'].append(min([track_tools.dist(CoMs1[j], CoM_in_next_image) for CoM_in_next_image in CoMs2]))


## define paralysis
MAXIMUM_mm_MOVEMENT = 0.25
plate_diam_mm = 49
plate_diam_px = []
PIX_PER_MM = 3020 / 49
MIMIMUM_SCORE = 0.9
for dataset in datasets:
    results = inference_results[dataset.root]
    for img in results.values():
        img['paralyzed'] =  [(distance / PIX_PER_MM) < MAXIMUM_mm_MOVEMENT for distance in img['min_dist_px']]
        if len(img['paralyzed']) > 0:
            passing_paralayzed = [i for (i, s) in zip(img['paralyzed'], img['scores']) if s > MIMIMUM_SCORE]
            img['pct_paralyzed'] =  sum(passing_paralayzed) / len(passing_paralayzed)
        else:
            img['pct_paralyzed'] = None

process_time = datetime.now()
## Draw tracking on images
## Just invert back to the 'expected' worm-dark
for dataset in datasets:
    results = inference_results[dataset.root]
    for img_path, inference in results.items():
        image = autocontrast(invert(inference['Image']), cutoff=0.8, ignore=None)
        image = np.expand_dims(np.array(image), 0)
        boxes = inference['boxes'][:,[1,0,3,2]]
        classes = []
        for i in range(len(inference['scores'])):
            if inference['scores'][i] < MIMIMUM_SCORE:
                classes.append(1)
            elif len(inference['paralyzed']) == 0:
                classes.append(4)
            elif inference['paralyzed'][i]:
                classes.append(2)
            else:
                classes.append(3)
        scores = inference['scores']
        category_index = {1:'low_score', 2: 'paralyzed', 3:'mobile', 4:'untracked'}
        class_to_color_map = {1:'grey', 2:'red', 3:'green', 4:'yellow'}
        
        annotated_image = vis_utils.draw_bounding_boxes_on_image_tensors(image,
                                                 boxes,
                                                 classes,
                                                 scores,
                                                 category_index,
                                                 max_boxes_to_draw=100,
                                                 min_score_thresh=0.2,
                                                 line_thickness=2,
                                                 skip_scores=False,
                                                 skip_labels=True,
                                                 use_normalized_coordinates=False,
                                                 class_to_color_map=class_to_color_map)
        
        w_tracking = os.path.join(dataset.root, 'w_tracking')
        utils.mkdir(w_tracking)
        Image.fromarray(annotated_image.numpy()).save(os.path.join(w_tracking, os.path.basename(img_path).replace('.tif', '.jpeg')))



## Write CSV paralysis data
df_aggr = None
for dataset in datasets:
    results = inference_results[dataset.root]
    df_indy = pd.DataFrame([{'image': k, 'pct_paralyzed': v['pct_paralyzed']} for k, v in results.items()])
    df_indy.to_csv(os.path.join(dataset.root,'paralysis.csv'))
    if df_aggr is None:
        df_aggr = pd.DataFrame([{'image': os.path.basename(k).replace('.' + file_type, ''), dataset.root: v['pct_paralyzed']} for k, v in results.items()])
    else:
        df_aggr = pd.concat([df_aggr.reset_index(drop=True), pd.DataFrame([{dataset.root: v['pct_paralyzed']} for v in results.values()])], axis=1)

df_aggr.to_csv(os.path.join(PATH_TO_DATA_FOLDERS,'paralysis.csv'))

write_time = datetime.now()
print(inf_time-start , process_time-start, write_time-start )


