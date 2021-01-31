import glob

import cv2
import numpy as np
import torch
from os.path import exists

import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image

from torch.utils.data.dataset import T_co


def find_boxes(image, klass=0):
    if image is None:
        return {}
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (image>0).astype(np.uint8))
#     print(labels.shape, labels.max())
    stats[:,2] += stats[:,0]
    stats[:,3] += stats[:,1]
    stats[:,[0,1,2,3]] = stats[:,[1,0,3,2]]
    stats = stats[:, :-1]
    boxes = stats[1:]
    masks = np.zeros((len(boxes), *image.shape), dtype=np.uint8)
    for i in range(len(boxes)):
        masks[i, boxes[i, 0]:boxes[i, 2], boxes[i, 1]:boxes[i, 3]] = (
                    labels[boxes[i, 0]:boxes[i, 2], boxes[i, 1]:boxes[i, 3]] == (i + 1))
    boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]
    klass = np.array([klass] * len(boxes))
    return dict(
        boxes=torch.from_numpy(boxes.astype(np.float32)),
        masks=torch.from_numpy(masks),
        labels=torch.from_numpy(klass.astype(np.int64)))


class LesionSegMask(Dataset):
    def __init__(self, split=None, root=None):
        if root is None:
            raise Exception("data root directory is none")
        images = os.listdir(root)
        images = [int(i[:2]) for i in images]
        self.max_index = max(images)
        self.images = []
        i = 1
        while os.path.exists(f'{root}/{i:05d}.jpg'):
            self.images.append((
                    f'{root}/{i:05d}.jpg',
                    f'{root}/{i:05d}_EX.tif',
                    f'{root}/{i:05d}_HE.tif',
                    f'{root}/{i:05d}_MA.tif',
                    f'{root}/{i:05d}_SE.tif',
                ))
            i += 1
        split_point = int(len(self.images) * 0.8)
        if split == 'train':
            self.images = self.images[:split_point]
        elif split == 'test':
            self.images = self.images[split_point:]
        else:
            raise NotImplementedError(f"the split name {split} not supported")
        self.ratio = 0.5

    def __getitem__(self, index):
        i = self.images[index]
        image = cv2.imread(i[0], cv2.IMREAD_COLOR)
        dsize = (int(image.shape[1] * self.ratio), int(image.shape[0] * self.ratio))
        image = cv2.resize(image, dsize=dsize)
        label = {}
        for cls, path in enumerate(i[1:]):
            if not exists(path):
                continue
            mask_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, dsize=dsize)
            res = find_boxes(mask_img, cls)
            for k, v in res.items():
                if k in label:
                    label[k] = torch.cat((label[k], v))
                else:
                    label[k] = v
        return image, label

    def __len__(self):
        return len(self.images)

class ClassificationTest(Dataset):
    def __init__(self, root:str, **kwargs):
        assert(os.path.exists(root))
        self.root = root
        if not root.endswith('/'):
            root = root + '/'
        root = root + '*'
        folders = glob.glob(root)
        classes = [os.path.split(i)[-1] for i in folders]
        self.records = []
        for f,c in zip(folders, classes):
            c = int(c)
            for file in glob.glob(f'{f}/*'):
                self.records.append(dict(
                    gt_label=c,
                    img_prefix=None,
                    img_info=dict(
                        filename=os.path.join(f, file),
                    )
                ))
        self.records = sorted(self.records, key=lambda k: k['img_info']['filename'])
        for i, rec in enumerate(self.records):
            rec['index'] = i
        super().__init__()

    def __getitem__(self, index):
        image = self.records[index]['img_info']['filename']
        image = Image.open(image)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(448),
            torchvision.transforms.CenterCrop(448),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ])
        image = transform(image)
        return image, self.records[index]['gt_label']

    def __len__(self):
        return len(self.records)

