import os
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import glob
import xml.etree.ElementTree as ET
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import datetime


class ImgsAndBBs(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.BBs = list(sorted(glob.glob(root + '/*.xml')))
        self.imgs = [xml_path.replace('.xml','.jpeg') for xml_path in self.BBs]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        BB_path = self.BBs[idx]
        #### mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
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
        image_id = torch.tensor([idx])
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

    def __len__(self):
        return len(self.imgs)




def get_instance_BB_model(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _run(model, optimizer, data_loader, data_loader_test, device, lr_scheduler, run, num_epochs=20):
    run_path = os.path.join('runs/' +datetime.datetime.now().strftime('%y-%m-%d_%H-%M')+ str(run))
    writer = SummaryWriter(run_path)
    print('Start training...')
    try:
        epoch = 0
        while epoch != num_epochs:
            # train for one epoch, printing every 10 iterations
            meters = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

            # update the learning rate
            lr_scheduler.step(meters['loss'].value)
            # evaluate on the test dataset
            coco_evaluator = evaluate(model, data_loader_test, device=device, writer=writer, global_steps=epoch*len(data_loader), meters=meters)
            ## maybe at this point I'd checkpoint the model? https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
            epoch = epoch + 1
    except KeyboardInterrupt:
        print('KeyboardInterrupt at epoch ' + str(epoch))
    finally:
        print("Closing up shop.")
        torch.save(model.state_dict(), os.path.join(run_path, 'trained_model.pth'))
        writer.close()
        return model, coco_evaluator

    
    
def main(run='_LRplat', num_epochs=150, ratio=0.9, batch_size=4, num_workers=2):
    print('Initializing...')
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # our dataset has two classes only - background and worm
    num_classes = 2
    # use our dataset and defined transformations
    print('Loading dataset...')
    dataset = ImgsAndBBs(r'F:\Hujber\PyTorch\workspace\wormLearn\images\all\jpeg', utils.get_transform(train=True))
    dataset_test = ImgsAndBBs(r'F:\Hujber\PyTorch\workspace\wormLearn\images\all\jpeg', utils.get_transform(train=False))
    
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    split = 1-round(len(indices) * ratio)
    dataset = torch.utils.data.Subset(dataset, indices[:-split])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-split:])
    
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=utils.collate_fn)
    
    # get the model using our helper function
    print('Setting up model...')
    model = get_instance_BB_model(num_classes)

    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.8)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)    ## Works a lot better than stepLR
    ## train
    m, e = _run(model, optimizer, data_loader, data_loader_test, device, lr_scheduler, run, num_epochs)
    


if __name__ == "__main__":
    main()