import os
import numpy as np
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import glob



class ImgsAndBBs(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(glob.glob(root + '/*.jpeg')))
        self.BBs = list(sorted(glob.glob(root + '/*.xml')))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        BB_path = self.BBs[idx]
        #### mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        #### mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        #### mask = np.array(mask)
        # instances are encoded as different colors
        #### obj_ids = np.unique(mask)
        # first id is the background, so remove it
        #### obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        #### masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        #### 


        boxes = []
        tree = ET.parse(BB_path)
        root = tree.getroot()
        for member in root.findall('object'):
            xmin = int(member[4][0].text)
            xmax = int(member[4][1].text)
            ymin = int(member[4][2].text)
            ymax = int(member[4][3].text)
            boxes.append([xmin, ymin, xmax, ymax])

        num_objs = len(boxes)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        #### masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

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

    def __len__(self):
        return len(self.imgs)