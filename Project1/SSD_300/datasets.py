import torch
from torch.utils.data import Dataset
import json
import os
# from PIL import Image
from PIL import Image,ImageFont, ImageDraw, ImageEnhance
from utils import transform
import numpy as np


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)
        self.images =self.images[:50]
        self.objects =self.objects[:50]


        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        # print(i, self.images[i])
        # exit()
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')
        print("image size ", image.size)

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        print("objects ",objects)
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each


class ADEDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        # with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
        #     self.images = json.load(j)
        # with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
        #     self.objects = json.load(j)
        # self.images = self.images[:50]
        # self.objects = self.objects[:50]

        self.images = os.listdir(os.path.join(data_folder,"Chair_ADE20K_IMG"))
        self.objects = os.listdir(os.path.join(data_folder,"Chair_ADE20K_MASK"))

        # self.images = self.images[:50]
        # self.objects = self.objects[:50]

        # for i in range(len(self.images)):
        #     path = os.path.join(self.data_folder,"Chair_ADE20K_MASK",self.images[i].replace(".jpg", "_seg.png"))
        #     if not os.path.exists(path):
        #         print(path)
        #         os.remove(os.path.join(self.data_folder,"Chair_ADE20K_IMG",self.images[i]))

        print("self.images ",self.images[:5], len(self.images))
        print("self.objects ",self.objects[:5], len(self.objects))
        # exit()
        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        # print(i, self.images[i])
        # exit()
        image = Image.open(os.path.join(self.data_folder,"Chair_ADE20K_IMG",self.images[i]), mode='r')
        image = image.convert('RGB')


        mask = Image.open(os.path.join(self.data_folder,"Chair_ADE20K_MASK",self.images[i].replace(".jpg", "_seg.png")))
        mask = np.array(mask)

        
        obj_ids = np.unique(mask)



        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        # print("mask 0 :",mask.shape , np.unique(mask))
        masks = mask == obj_ids[:, None, None]

        # print("mask 1 :",mask.shape , np.unique(mask))
        # exit()

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        labels=[]
        difficulties=[]

        # draw = ImageDraw.Draw(image)

        # print("num_objs ",num_objs)

        for k in range(num_objs):
            pos = np.where(masks[k])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)
            difficulties.append(0)
        if num_objs == 0:
            boxes = [[0,0,1,1]]
            labels =[0]
            difficulties = [0]

            # draw.rectangle(((xmin,ymin),(xmax, ymax)), fill=None,outline ="blue",width=2)
        # image.save(str(k)+"rgb.png")
        # print("boxes ",boxes)
        # exit()

        # Read objects in this image (bounding boxes, labels, difficulties)
        # objects = self.objects[i]
        # print("objects ",objects)
        boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
        labels = torch.LongTensor(labels)  # (n_objects)
        difficulties = torch.ByteTensor(difficulties)  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
