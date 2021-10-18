from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import os
import re
import torch
import glob
import datetime
import PIL.Image as Image
from PIL.ImageOps import invert
import numpy as np
from cv2 import cv2
import xml.etree.ElementTree as ET



from . import transforms as T



#######
###   Copied directly from lableImg https://github.com/tzutalin/labelImg
###    Only removing dependencies on other libs in that package.
#######


XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'

class PascalVocWriter:

    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = str(each_object['name'])
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            if int(float(each_object['ymax'])) == int(float(self.imgSize[0])) or (int(float(each_object['ymin']))== 1):
                truncated.text = "1" # max == height or min
            elif (int(float(each_object['xmax']))==int(float(self.imgSize[1]))) or (int(float(each_object['xmin']))== 1):
                truncated.text = "1" # max == width or min
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str( bool(each_object['difficult']) & 1 )
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class PascalVocReader:

    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        try:
            self.parseXML()
        except:
            pass

    def getShapes(self):
        return self.shapes

    def addShape(self, label, bndbox, difficult):
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None, difficult))

    def parseXML(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            label = object_iter.find('name').text
            # Add chris
            difficult = False
            if object_iter.find('difficult') is not None:
                difficult = bool(int(object_iter.find('difficult').text))
            self.addShape(label, bndbox, difficult)
        return True
        


class LabelFileError(Exception):
    pass


class LabelFile(object):
    # It might be changed as window creates. By default, using XML ext
    # suffix = '.lif'
    suffix = XML_EXT

    def __init__(self, filename=None):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        self.verified = False

    def savePascalVocFormat(self, filename, shapes, imagePath, imageShape):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        #imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        # Read from file path because self.imageData might be empty if saving to
        # Pascal format
        writer = PascalVocWriter(imgFolderName, imgFileName,
                                 imageShape, localImgPath=imagePath)
        writer.verified = self.verified

        for shape in shapes:
            points = shape['points']
            label = shape['label']
            # Add Chris
            difficult = int(shape['difficult'])
            bndbox = LabelFile.convertPoints2BndBox(points)
            writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3], label, difficult)

        writer.save(targetFile=filename)
        return

    def toggleVerify(self):
        self.verified = not self.verified


    @staticmethod
    def isLabelFile(filename):
        fileSuffix = os.path.splitext(filename)[1].lower()
        return fileSuffix == LabelFile.suffix

    @staticmethod
    def convertPoints2BndBox(points):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1

        if ymin < 1:
            ymin = 1

        return (int(xmin), int(ymin), int(xmax), int(ymax))

        
#######
###   End direct copy from lableImg https://github.com/tzutalin/labelImg
#######


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


class Imageset(torch.utils.data.Dataset):
    def __init__(self, root, parent_path=None, transforms=None, img_type='.tif', bb_type=None, from_subfolders_idx=None):  # bb_type='.xml'
        '''
        from_subfolders_idx ---  indices of which subfolders to pull images from. Used when imagesets are split train/test. Ignored if None
        '''
        self.root = root
        self.root_stub = root.replace(parent_path, '') if parent_path is not None else None
        self.transforms = transforms
        # maybe add some smarter filter to ensure just the appropriate images are selected?
        # self.img_paths = list(sorted(glob.glob(root + '/*' + img_type)))
        # self.ts_fmt = ts_fmt
        # self.zero = datetime.datetime.strptime('00', r'%S').strftime(ts_fmt)
        slashstar = '/*'
        img_type = [img_type] if type(img_type) is not list else img_type
        self.BBs = None if bb_type is None else []
        self.imgs = []
        if from_subfolders_idx:
            for subfolder in [glob.glob(root + '/*')[i] for i in from_subfolders_idx]:
                if bb_type:
                    n_new_bbs, n_new_imgs = self.get_BBs_and_img_paths(subfolder, bb_type, img_type)
                    print(str(n_new_bbs) + ' ' + bb_type + ' and ' + str(n_new_imgs) + ' ' + ' or '.join(img_type) + ' files found in ' + subfolder)
                else:
                    self.get_img_paths(subfolder, img_type)
                    print(str(n_new_imgs) + ' ' + ' or '.join(img_type) + ' files found in ' + subfolder)
            print('TOTAL:  ' + str(len(self.BBs)) + ' ' + bb_type + ' and ' + str(len(self.imgs)) + ' ' + ' or '.join(img_type) + ' files found in ' + str(len(from_subfolders_idx)) + ' of ' + str(len(glob.glob(root + '/*'))) + ' subfolders in '+ root)
        else:
            if bb_type:
                n_new_bbs, n_new_imgs = self.get_BBs_and_img_paths(root, bb_type, img_type)
                print(str(len(self.BBs)) + ' ' + bb_type + ' and ' + str(len(self.imgs)) + ' ' + ' or '.join(img_type) + ' files found in ' + root)
            else:
                self.get_img_paths(root, img_type)
                print(str(len(self.imgs)) + ' ' + ' or '.join(img_type) + ' files found in ' + root)
 
    def __repr__(self):
        return f'Imageset at {self.root_stub}'
       
    def get_BBs_and_img_paths(self, path, bb_type, img_type):
        new_BBs = list(sorted(glob.glob(path + '/*' + bb_type)))
        # print('Found in ', path)
        # print(new_BBs)
        n_new_imgs = 0
        self.BBs.extend(new_BBs)
        for xml_path in new_BBs:
            for itype in img_type:
                img_path = xml_path.replace(bb_type, itype)
                if os.path.exists(img_path):
                    self.imgs.append(img_path)
                    n_new_imgs = n_new_imgs + 1
        # print(len(new_BBs), n_new_imgs)
        return len(new_BBs), n_new_imgs
    
    def get_img_paths(self, path, img_type):
        for ext in img_type:
           self.imgs.extend(glob.glob(path + '/*' + ext))             
        # self.imgs = list(sorted(glob.glob(root + slashstar + img_type)))
        
        
    ###### this init version has subfolders_too, but i think i never use it and it complicated implemantation other things, so im removing it for now
    # def __init__(self, root, parent_path, transforms, img_type='.tif', bb_type=None, subfolders_too=False):  # bb_type='.xml'
    #     self.root = root
    #     self.root_stub = root.replace(parent_path, '')
    #     self.transforms = transforms
    #     # maybe add some smarter filter to ensure just the appropriate images are selected?
    #     # self.img_paths = list(sorted(glob.glob(root + '/*' + img_type)))
    #     # self.ts_fmt = ts_fmt
    #     # self.zero = datetime.datetime.strptime('00', r'%S').strftime(ts_fmt)
    #     slashstar = '/*/*' if subfolders_too else '/*'
    #     img_type = [img_type] if type(img_type) is not list else img_type
    #     self.BBs = None
    #     if bb_type:
    #         self.BBs = list(sorted(glob.glob(root + slashstar + bb_type)))
    #         self.imgs = []
    #         for xml_path in self.BBs:
    #             for itype in img_type:
    #                 img_path = xml_path.replace(bb_type, itype)
    #                 if os.path.exists(img_path):
    #                     self.imgs.append(img_path)
    #         print(str(len(self.BBs)) + ' ' + bb_type + ' and ' + str(len(self.imgs)) + ' ' + ' or '.join(img_type) + ' files found in ' + root)
    #     else:
    #         self.imgs = []
    #         for ext in img_type:
    #            self.imgs.extend(glob.glob(root + slashstar + ext))             
    #         # self.imgs = list(sorted(glob.glob(root + slashstar + img_type)))
    #         print(str(len(self.imgs)) + ' ' + ' or '.join(img_type) + ' files found in ' + root)

    def __getitem__(self, idx):
        # load images and bbs
        image_id = torch.tensor([idx])
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        if self.BBs:
            BB_path = self.BBs[idx]
            #print(BB_path)
            boxes = []
            tree = ET.parse(BB_path)
            root = tree.getroot()
            for member in root.findall('object'):
                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)
                box = [xmin, ymin, xmax, ymax]
                # print(box)
                boxes.append(box)
            num_objs = len(boxes)
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            #### masks = torch.as_tensor(masks, dtype=torch.uint8)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            #
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            #### target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target
        if self.transforms is not None:
            img = self.transforms(img)
        return img, img_path

    def __len__(self):
        return len(self.imgs)
    
    def get_by_name(self, name):
        return Image.open(name).convert("RGB")
    
    def get_by_name_as_gray_Image(self, name):
        return Image.open(name).convert("L")


class ImageStackset(torch.utils.data.Dataset):
    def __init__(self, root, parent_path, transforms, ts_fmt, file_type='.tif'):
        self.root = root
        self.root_stub = root.replace(parent_path, '')
        self.transforms = transforms
        # maybe add some smarter filter to ensure just the appropriate images are selected?
        self.img_paths = list(sorted(glob.glob(root + '/*' + file_type)))
        self.imgs = None
        self.ts_fmt = ts_fmt
        self.zero = datetime.datetime.strptime('00', r'%S').strftime(ts_fmt)
        print(str(len(self.img_paths)) + ' ' + file_type + ' files found in ' + root)
        self.load_and_process_stack()
    
    def __getitem__(self, idx):
        # load image
        img_array = self.imgs['all'][self.img_paths[idx]]
        img = Image.fromarray(img_array)
        # image_id = torch.tensor([idx])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.img_paths[idx]
    
    def __len__(self):
        return len(self.img_paths)

    def load_and_process_stack(self):
        img_paths = sorted(self.img_paths)
        
        ## open and store inverted img as np array. 
        ## img_dic['all'] is a dic where each {key: value} pairs is {path: np.array}
        ## img_dic['00_00'] hold the str name of the 00_00 img path.
        img_dic = {'all': {}}
        for path in img_paths:
            img = Image.open(path).convert("RGB") 
            inverted_img = invert(img)
            img_dic['all'][path] = np.array(inverted_img)
            if self.zero in self.get_timestamp_from_name(path):
                img_dic['00_00'] = path
        
        self.align_stack(img_dic) ## alignment happens in place, no need to reassign
        self.normalize_stack_intensity(img_dic)  ## normalization happens in place, no need to reassign
        first = img_dic['all'][img_dic['00_00']]
        for i, im in img_dic['all'].items():
            if i == img_dic['00_00']:
                continue
            self.img_minus_first(im, first)
        self.to_uint8(img_dic)
        self.imgs = img_dic
    
    def get_timestamp_from_name(self, name):
        ts_regex_str = self.ts_fmt.replace(r'%d', r'\d{2}').replace(r'%H', r'\d{2}').replace(r'%M', r'\d{2}').replace(r'%S', r'\d{2}')
        regex_overlappable = '(?=(' + ts_regex_str + '))'
        filebase = os.path.splitext(os.path.basename(name))[0]
        name_no_serial = '_'.join(filebase.split('_')[0:-1])
        return re.findall(regex_overlappable, name_no_serial)[-1]
    
    def to_uint8(self, img_dic):
        for i, im in img_dic['all'].items():
            np.clip(im, 0, 255, im)
            img_dic['all'][i] = im.astype(np.uint8)
    
    ##### ALIGNMENT ON CORNERS
    def align_stack(self, stack_dic, trim_ratio=0.18, pad_pix=10):
        image_h, image_w = stack_dic['all'][stack_dic['00_00']].shape[:2]   ## Images are HWC
        trim_h = round(image_h * trim_ratio)
        trim_w = round(image_w * trim_ratio)
        warp_mode = cv2.MOTION_TRANSLATION
        number_of_iterations = 100
        termination_eps = 1e-7
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                    number_of_iterations, termination_eps)
        
        im1 = stack_dic['all'][stack_dic['00_00']]
        TL1 = im1[pad_pix:trim_h, pad_pix:trim_w, 0]
        BL1 = im1[-trim_h:-pad_pix, pad_pix:trim_w, 0]
        TR1 = im1[pad_pix:trim_h, -trim_w:-pad_pix, 0]
        BR1 = im1[-trim_h:-pad_pix, -trim_w:-pad_pix, 0]
        corners1 = [TL1, BL1, TR1, BR1]
        
        for i, im2 in stack_dic['all'].items():
            if i == stack_dic['00_00']:
                continue
            
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
            
            stack_dic['all'][i] = np.stack([aligned_image0, aligned_image1, aligned_image2], axis=2)
        # return stack #  I think this all happens in place now so no return is necessary
    
    def normalize_stack_intensity(self, stack_dic, pad_pix=20):        
        f1 = stack_dic['all'][stack_dic['00_00']].astype('float64')
        for i, im2 in stack_dic['all'].items():
            if i == stack_dic['00_00']:
                continue
            f2 = im2.astype('float64')
            ratio = np.mean(f1[pad_pix:-pad_pix, pad_pix:-pad_pix, :]) / np.mean(f2[pad_pix:-pad_pix, pad_pix:-pad_pix, :])
            stack_dic['all'][i] = f2 * ratio
        stack_dic['all'][stack_dic['00_00']] = f1
        # return stack #  I think this all happens in place now so no return is necessary
    
    def img_minus_first(self, img, first, ch=1):
        '''
        Arguments:
        img: HWC nump array
        first: HWC nump array
        
        Returns:
        a HWC nump array, with img[:,:,ch] - first [:,:,ch]
        '''
        img[:,:,ch] -= first[:,:,ch]
    
    def get_by_name(self, name):
        return self.imgs['all'][name]
    
    def get_by_name_as_gray_Image(self, name):
        return Image.fromarray(self.get_by_name(name)[:, :, 0])

        

class DatasetReader(object):
    def __init__(self, loader):
        self.loader = loader
        
    def serve_images(self):
        for image in self.loader:
            yield image


class PascalVOCReader(PascalVocReader):
    def getInference(self):
        return dict(boxes=self.getBoxes(),
                    labels=self.getLabels(),
                    scores=self.getScores())
    
    def getBoxes(self):
        return np.array([self.points_to_bbox(shape) for shape in self.shapes])
    
    def getLabels(self):
        return np.ones(len(self.shapes), dtype='uint64')
        
    def getScores(self):
        return np.array([1 if shape[0] == 'worm' else shape[0] for shape in self.shapes], dtype='float32')
    
    def points_to_bbox(self, shape):
        x1 = min(p[0] for p in shape[1])
        y1 = min(p[1] for p in shape[1])
        x2 = max(p[0] for p in shape[1])
        y2 = max(p[1] for p in shape[1])
        return np.array([x1, y1, x2, y2], dtype='float32')



def inference_transforms(invert=False):
    transforms = []
    transforms.append(T.ToTensor())
    if invert:
        transforms.append(T.Invert())
    return T.Compose(transforms)

