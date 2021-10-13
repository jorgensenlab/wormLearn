import json
import tempfile

import numpy as np
import copy
import time
import torch
import torch._six

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from collections import defaultdict

from . import utils
from . import vis_tools


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_dt = loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()
            
    ######################################################################################################
    def summary_to_tensorboard(self, writer, global_steps, meters):
        # print('here i send numbers to TensorBoard')
        # print('for example, ')
        for iou_type, coco_eval in self.coco_eval.items():
            writer.add_scalar('CocoMetric.....mean_Average_Precision/IoU=0.50:0.95_area=all_maxDets=100', coco_eval.stats[0], global_steps)
            writer.add_scalar('CocoMetric.....mean_Average_Precision/IoU=0.50_area=all_maxDets=100', coco_eval.stats[1], global_steps)
            writer.add_scalar('CocoMetric.....mean_Average_Precision/IoU=0.75_area=all_maxDets=100', coco_eval.stats[2], global_steps)
            writer.add_scalar('CocoMetric.....mean_Average_Precision/IoU=0.50:0.95_area=small_maxDets=100', coco_eval.stats[3], global_steps)
            writer.add_scalar('CocoMetric.....mean_Average_Precision/IoU=0.50:0.95_area=medium_maxDets=100', coco_eval.stats[4], global_steps)
            writer.add_scalar('CocoMetric.....mean_Average_Recall/IoU=0.50:0.95_area=all_maxDets=1', coco_eval.stats[6], global_steps)
            writer.add_scalar('CocoMetric.....mean_Average_Recall/IoU=0.50:0.95_area=all_maxDets=10', coco_eval.stats[7], global_steps)
            writer.add_scalar('CocoMetric.....mean_Average_Recall/IoU=0.50:0.95_area=all_maxDets=100', coco_eval.stats[8], global_steps)
            writer.add_scalar('CocoMetric.....mean_Average_Recall/IoU=0.50:0.95_area=small_maxDets=100', coco_eval.stats[9], global_steps)
            writer.add_scalar('CocoMetric.....mean_Average_Recall/IoU=0.50:0.95_area=medium_maxDets=100', coco_eval.stats[10],global_steps)
            # print("IoU metric: {}".format(iou_type))
            # print(coco_eval.stats)
        for meter, value in meters.items():
            writer.add_scalar('Meters/'+str(meter), value.median, global_steps)
        self.detectionleft_groundtruthright(writer, global_steps)
        writer.flush()
    
    def detectionleft_groundtruthright(self, writer, global_steps):
        '''
        #### e = self
        #### (37,1) i tihnk referes to the image_id = 37. 1 = catID
        >>> e.coco_eval['bbox']._gts[(37,1)][0]
        {'image_id': 37, 'bbox': [1120.0, 5.0, 16.0, 32.0], 'category_id': 1, 'area': 512.0, 'iscrowd': 0, 'id': 804, 'ignore': 0, '_ignore': 1}
        >>> e.coco_eval['bbox']._gts[(37,1)][1]
        {'image_id': 37, 'bbox': [982.0, 208.0, 25.0, 25.0], 'category_id': 1, 'area': 625.0, 'iscrowd': 0, 'id': 805, 'ignore': 0, '_ignore': 1}
        >>>
        
        
        >>> e.coco_eval['bbox']._dts[(37,1)][0]
        {'image_id': 37, 'category_id': 1, 'bbox': [1150.638427734375, 725.9815063476562, 24.51025390625, 26.27838134765625], 'score': 0.9767993092536926, 'segmentation': [[1150.638427734375, 725.9815063476562, 1150.638427734375, 752.2598876953125, 1175.148681640625, 752.2598876953125, 1175.148681640625, 725.9815063476562]], 'area': 644.0897990763187, 'id': 1, 'iscrowd': 0}
        >>> e.coco_eval['bbox']._dts[(37,1)][1]
        {'image_id': 37, 'category_id': 1, 'bbox': [980.4163818359375, 208.55953979492188, 27.1900634765625, 25.393295288085938], 'score': 0.9662157893180847, 'segmentation': [[980.4163818359375, 208.55953979492188, 980.4163818359375, 233.9528350830078, 1007.6064453125, 233.9528350830078, 1007.6064453125, 208.55953979492188]], 'area': 690.4453107621521, 'id': 2, 'iscrowd': 0}
        >>>
        
        
        >>> e.coco_gt.dataset['images'][0].keys()
        dict_keys(['id', 'height', 'width', 'img'])
        e.coco_gt.dataset['images'][0]['img']
        >>> e.coco_gt.dataset['images'][0]['img'].shape
            torch.Size([3, 1952, 2592])    == CHW
        
          ### _gt == {'image_id': 37,
                    # 'bbox': [1120.0, 5.0, 16.0, 32.0], 
                    # 'category_id': 1,
                    # 'area': 512.0,
                    # 'iscrowd': 0,
                    # 'id': 804,
                    # 'ignore': 0,
                    # '_ignore': 1}
  
  
          ### _dt == {'image_id': 37,
                    # 'category_id': 1, 
                    # 'bbox': [1150.638427734375, 725.9815063476562, 24.51025390625, 26.27838134765625],
                    # 'score': 0.9767993092536926,
                    # 'segmentation': [[1150.638427734375, 725.9815063476562, 1150.638427734375, 752.2598876953125, 1175.148681640625, 752.2598876953125, 1175.148681640625, 725.9815063476562]],
                    # 'area': 644.0897990763187,
                    # 'id': 1,
                    # 'iscrowd': 0}
        
        
        
        One "img,gt,dt" dictionary per image in this list:
        eval_dict_list = list({'img':<img>,
                     'gt': {
                          'bbox': [[4],[4],...],
                          'category_id': [<int>,<int>,...]
                          },
                     'dt': {
                          'bbox': [[4],[4],...],
                          'category_id': [<int>,<int>,...],
                          'score': [<float>,<float>,...]
                          }
                     })
        '''
        def XYHW_to_X1Y1X2Y2(box):
            npbox = np.array(box)
            npbox[2:] += npbox[:2]
            npbox = npbox[[1,0,3,2]]
            return list(npbox)
        
        im_dict = {}
        for im in self.coco_gt.dataset['images']:
            im_dict[im['id']] = im['img']
            
        gt_img_IDs = [k[0] for k in self.coco_eval['bbox']._gts.keys()]
        #print('_gts.keys ', self.coco_eval['bbox']._gts.keys())
        eval_dict_list = []
        for imgID in gt_img_IDs:
            eval_dict = {}
            
            eval_dict['img'] = im_dict[imgID]
            
            eval_dict['gt'] = {'bbox':[], 'category_id': []}
            for box in self.coco_eval['bbox']._gts[(imgID,1)]:
                eval_dict['gt']['bbox'].append(XYHW_to_X1Y1X2Y2(box['bbox']))
                eval_dict['gt']['category_id'].append(box['category_id'])
                
            eval_dict['dt'] = {'bbox':[], 'category_id': [], 'score': []}
            for box in self.coco_eval['bbox']._dts[(imgID,1)]:
                eval_dict['dt']['bbox'].append(XYHW_to_X1Y1X2Y2(box['bbox']))
                eval_dict['dt']['category_id'].append(box['category_id'])
                eval_dict['dt']['score'].append(box['score'])
                
            eval_dict_list.append(eval_dict)
        DLGR = vis_tools.draw_superimposed_evaluation_image(eval_dict_list,
                                                           category_index = {1:'worm'},
                                                           max_boxes_to_draw=100,
                                                           min_score_thresh=0.2,
                                                           use_normalized_coordinates=False)         ## returns list of image tensor
        #print(len(DLGR), ' DLGR')
        img_tensor = DLGR[0]  ## just one for now.
        writer.add_image("Superimposed", img_tensor, global_step=global_steps, walltime=None, dataformats='HWC')


    ########################################################################################################

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################

# Ideally, pycocotools wouldn't have hard-coded prints
# so that we could avoid copy-pasting those two functions

def createIndex(self):
    # create index
    # print('creating index...')
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if 'annotations' in self.dataset:
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

    if 'images' in self.dataset:
        for img in self.dataset['images']:
            imgs[img['id']] = img

    if 'categories' in self.dataset:
        for cat in self.dataset['categories']:
            cats[cat['id']] = cat

    if 'annotations' in self.dataset and 'categories' in self.dataset:
        for ann in self.dataset['annotations']:
            catToImgs[ann['category_id']].append(ann['image_id'])

    # print('index created!')

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats


maskUtils = mask_util


def loadRes(self, resFile):
    """
    Load result file and return a result api object.
    :param   resFile (str)     : file name of result file
    :return: res (obj)         : result api object
    """
    res = COCO()
    res.dataset['images'] = [img for img in self.dataset['images']]

    # print('Loading and preparing results...')
    # tic = time.time()
    if isinstance(resFile, torch._six.string_classes):
        anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:
        anns = self.loadNumpyAnnotations(resFile)
    else:
        anns = resFile
    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
        'Results do not correspond to current coco set'
    if 'caption' in anns[0]:
        imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
        res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
        for id, ann in enumerate(anns):
            ann['id'] = id + 1
    elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if 'segmentation' not in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['area'] = bb[2] * bb[3]
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'segmentation' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            # now only support compressed RLE format as segmentation results
            ann['area'] = maskUtils.area(ann['segmentation'])
            if 'bbox' not in ann:
                ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'keypoints' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            s = ann['keypoints']
            x = s[0::3]
            y = s[1::3]
            x1, x2, y1, y2 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann['area'] = (x2 - x1) * (y2 - y1)
            ann['id'] = id + 1
            ann['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    # print('DONE (t={:0.2f}s)'.format(time.time()- tic))

    res.dataset['annotations'] = anns
    createIndex(res)
    return res


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}
    # print(self.ious)
    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    # print(evalImgs)
    # 0/0
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
