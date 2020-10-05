import os
import numpy as np
import torch
from PIL import Image
import matplotlib


def load_sample_data(img_path):

    img = Image.open(img_path).convert('RGB')
    img_tensor, _ = get_transform(False)(img, None)
    img_tensor = img_tensor.cuda()
    x = [img_tensor]

    return x


class Chair_ADE_Dataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # print("self.transforms ",self.transforms )
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Chair_ADE20K_IMG"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Chair_ADE20K_MASK"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "Chair_ADE20K_IMG", self.imgs[idx])
        mask_path = os.path.join(self.root, "Chair_ADE20K_MASK", self.masks[idx])
        # print("mask_path ",mask_path)
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # matplotlib.image.imsave(os.path.join(self.root, "VS_PedMasks", self.masks[idx]), mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # print("obj_ids ",obj_ids)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            
        else:
            area = torch.as_tensor([], dtype=torch.float32)


        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)



class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # print("self.transforms ",self.transforms )
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        print("mask_path ",mask_path)
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        matplotlib.image.imsave(os.path.join(self.root, "VS_PedMasks", self.masks[idx]), mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # print("obj_ids ",obj_ids)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# def get_model_instance_segmentation(num_classes):
#     # load an instance segmentation model pre-trained pre-trained on COCO
#     model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

#     # get number of input features for the classifier
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     # replace the pre-trained head with a new one
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#     # now get the number of input features for the mask classifier
#     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#     hidden_layer = 256
#     # and replace the mask predictor with a new one
#     model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
#                                                        hidden_layer,
#                                                        num_classes)

#     return model


import transforms as T
import utils

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# dataset = PennFudanDataset('/vinai/vuonghn/Research/CV_courses/PennFudanPed', get_transform(train=True))
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,collate_fn=utils.collate_fn)
# # For Training
# images,targets = next(iter(data_loader))
# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]
# output = model(images,targets)   # Returns losses and detections
# # For inference
# model.eval()
# x = [torch.rand(3, 300, 400)]
# predictions = model(x)           # Returns predictions

# print("predictions ",predictions)
# exit()

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


from engine import train_one_epoch, evaluate
import utils


def train():

    TRAIN = False
    path_model = "/vinai/vuonghn/Research/CV_courses/Mask_RCNN_pytourch/logs_chair_ade/epoch_v1_170.pth"

    path_save = "/vinai/vuonghn/Research/CV_courses/Mask_RCNN_pytourch/logs_chair_ade/"
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = Chair_ADE_Dataset('/vinai/vuonghn/Research/CV_advanced_class/Dataset/Chair_ADE20K/training/', get_transform(train=True))
    dataset_test = Chair_ADE_Dataset('/vinai/vuonghn/Research/CV_advanced_class/Dataset/Chair_ADE20K/val/', get_transform(train=True))

    print("Train: ", len(dataset))
    print("Val: ", len(dataset_test))
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    indices_test = torch.randperm(len(dataset_test)).tolist()
    
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices_test[:10])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    # exit()

    # move model to the right device

    print("Prepare load model")
    if TRAIN == False:
        model = torch.load(path_model)
        evaluate(model, data_loader_test, device=device)
    
    

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 200

    for epoch in range(num_epochs):
        path_save_epoch = os.path.join(path_save,"epoch_v1_"+str(epoch)+".pth")

        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        torch.save(model, path_save_epoch)
        # path = "/vinai/vuonghn/Research/CV_courses/PennFudanPed/PNGImages/FudanPed00001.png"
        # img = load_sample_data(path)
        # print("out ", model(img))

train()



# path_img = "/vinai/vuonghn/Research/CV_courses/Dataset/Chair_ADE20K/training/Chair_ADE20K_IMG/"
# path_seg = "/vinai/vuonghn/Research/CV_courses/Dataset/Chair_ADE20K/training/Chair_ADE20K_MASK"
# def check_data(path_seg, path_img):
#     list_IDs = os.listdir(path_img)
#     for i,ID in enumerate(list_IDs):
#         print(i, "ID ", ID)
#         img_path = os.path.join(path_img,ID)
#         mask_path = os.path.join(path_seg,ID.replace(".jpg", "_seg.png"))
#         try:
#             # print("work ", ID)
#             img = Image.open(img_path).convert("RGB")
#             # note that we haven't converted the mask to RGB,
#             # because each color corresponds to a different instance
#             # with 0 being background
#             mask = Image.open(mask_path)
#             # convert the PIL Image into a numpy array
#             mask = np.array(mask)
#             # matplotlib.image.imsave(os.path.join(self.root, "VS_PedMasks", self.masks[idx]), mask)
#             # instances are encoded as different colors
#             obj_ids = np.unique(mask)
#             # print("obj_ids ",obj_ids)
#             # first id is the background, so remove it
#             obj_ids = obj_ids[1:]

#             # split the color-encoded mask into a set
#             # of binary masks
#             masks = mask == obj_ids[:, None, None]

#             # get bounding box coordinates for each mask
#             num_objs = len(obj_ids)
#             boxes = []
#             for i in range(num_objs):
#                 pos = np.where(masks[i])
#                 xmin = np.min(pos[1])
#                 xmax = np.max(pos[1])
#                 ymin = np.min(pos[0])
#                 ymax = np.max(pos[0])
#                 boxes.append([xmin, ymin, xmax, ymax])

#             # convert everything into a torch.Tensor
#             boxes = torch.as_tensor(boxes, dtype=torch.float32)
#             # there is only one class
#             labels = torch.ones((num_objs,), dtype=torch.int64)
#             masks = torch.as_tensor(masks, dtype=torch.uint8)

#             # image_id = torch.tensor([idx])
#             area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#             # suppose all instances are not crowd
#             iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

#             target = {}
#             target["boxes"] = boxes
#             target["labels"] = labels
#             target["masks"] = masks
#             # target["image_id"] = image_id
#             target["area"] = area
#             target["iscrowd"] = iscrowd
#         except:
#             print("NOTE WORK ",ID)
#             print("img_path ",img_path)
#             print("mask_path ",mask_path)
#             os.remove(img_path)
#             os.remove(mask_path)
#             continue

#             # exit()
# check_data(path_seg, path_img)



    
