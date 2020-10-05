import os
import numpy as np
import torch
from PIL import Image,ImageFont, ImageDraw, ImageEnhance
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import transforms as T
import utils
from engine import train_one_epoch, evaluate

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def convert_to_tensor(img):
    img_tensor, _ = get_transform(False)(img, None)
    img_tensor = img_tensor.cuda()
    x = [img_tensor]

    return x

def test_sample(img_path,mask_path,model):
    mask = Image.open(mask_path)
    mask = np.array(mask)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[1:]


    masks = mask == obj_ids[:, None, None]

    num_objs = len(obj_ids)
    boxes = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append(((xmin, ymin),( xmax, ymax)))

    img = Image.open(img_path).convert('RGB')

    img_tensor = convert_to_tensor(img)
    result = model(img_tensor)
    result_box = list(result[0]['boxes'])
    result_labels = list(result[0]['labels'])
    result_scores = list(result[0]['scores'])
    result_masks = list(result[0]['masks'])

    
    box_0 = ((int(result_box[0][0]),int(result_box[0][1])),(int(result_box[0][2]),int(result_box[0][3])))
    box_1 = ((int(result_box[1][0]),int(result_box[1][1])),(int(result_box[1][2]),int(result_box[1][3])))

    print(result_box)
    print(result_labels)
    print(result_scores)
    print(result_masks)
    draw = ImageDraw.Draw(img)
    draw.rectangle(boxes[0], fill=None,outline ="red",width=2)
    draw.rectangle(boxes[1], fill=None,outline ="red",width=2)

    draw.rectangle(box_0, fill=None,outline ="blue",width=2)
    draw.rectangle(box_1, fill=None,outline ="blue",width=2)


    # draw.rectangle(box_2, fill=None,outline ="blue",width=2)
    img.save("rgb.png")

img_path = "/vinai/vuonghn/Research/CV_courses/Dataset/Chair_ADE20K/val/Chair_ADE20K_IMG/ADE_val_00001198.jpg"
mask_path = "/vinai/vuonghn/Research/CV_courses/Dataset/Chair_ADE20K/val/Chair_ADE20K_MASK/ADE_val_00001198_seg.png"
model = torch.load("/vinai/vuonghn/Research/CV_courses/Mask_RCNN_pytourch/logs_chair_ade/epoch_v1_170.pth")
test_sample(img_path,mask_path,model)
exit()


img_path = "/vinai/vuonghn/Research/CV_courses/a-PyTorch-Tutorial-to-Object-Detection/VOCdevkit/VOC2007/JPEGImages/000005.jpg"
box = ((4, 243), (66, 373))
img = Image.open(img_path).convert('RGB')
draw = ImageDraw.Draw(img)
draw.rectangle(box, fill=None,outline ="blue",width=2)
img.save("rgb.png")