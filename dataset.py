import os
import torch
import numpy as np
from xml.dom import minidom
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm


class TrainDataset(Dataset):
    def __init__(self, root_dir='data', transform=None):
        classes = ['holothurian', 'echinus', 'scallop', 'starfish']

        self.transform = transform

        self.labelDic = {}
        self.labelDic['index2label'] = {}
        self.labelDic['label2index'] = {}
        for c in classes:
            self.labelDic['label2index'][c] = len(self.labelDic['label2index'])
            self.labelDic['index2label'][len(self.labelDic['index2label'])] = c

        self.num_classes = len(self.labelDic['label2index'])

        self.imagePath = os.path.join(root_dir, 'train/image')
        self.boxPath = os.path.join(root_dir, 'train/box')

        self.imgList = os.listdir(self.imagePath)
        self.boxList = [self.readXML(os.path.join(self.boxPath, imgName[:-4]+'.xml')) for imgName in self.imgList]

    def readXML(self, filename):
        xml_open = minidom.parse(filename)
        root =xml_open.documentElement
        objects = root.getElementsByTagName('object')
        output = []
        for ob in objects:
            name = ob.getElementsByTagName('name')
            bndbox = ob.getElementsByTagName('bndbox')
            xmin = bndbox[0].getElementsByTagName('xmin')
            ymin = bndbox[0].getElementsByTagName('ymin')
            xmax = bndbox[0].getElementsByTagName('xmax')
            ymax = bndbox[0].getElementsByTagName('ymax')

            name = name[0].firstChild.data
            xmin = xmin[0].firstChild.data
            ymin = ymin[0].firstChild.data
            xmax = xmax[0].firstChild.data
            ymax = ymax[0].firstChild.data
            if name not in self.labelDic['label2index']:
                continue
            output.append([int(xmin), int(ymin), int(xmax), int(ymax), self.labelDic['label2index'][name]])
        return output

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, index):
        imgName = self.imgList[index]
        img = cv2.imread(os.path.join(self.imagePath, imgName))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255

        boxes = self.boxList[index]
        annotations = np.zeros((0, 5))
        for box in boxes:
            annotation = np.zeros((1, 5))
            annotation[0, :] = box
            annotations = np.append(annotations, annotation, axis=0)
        
        sample = {'img': img, 'annot': annotations}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def label2index(self, label):
        return self.labelDic['label2index'][label]

    def index2label(self, index):
        return self.labelDic['index2label'][index]

    def getImageName(self, index):
        imgName = self.imgList[index]
        return imgName

class TestDataset(Dataset):
    def __init__(self, root_dir='data', transform=None):
        classes = ['holothurian', 'echinus', 'scallop', 'starfish']

        self.transform = transform

        self.labelDic = {}
        self.labelDic['index2label'] = {}
        self.labelDic['label2index'] = {}
        for c in classes:
            self.labelDic['label2index'][c] = len(self.labelDic['label2index'])
            self.labelDic['index2label'][len(self.labelDic['index2label'])] = c

        self.num_classes = len(self.labelDic['label2index'])

        self.imagePath = os.path.join(root_dir, 'test-A-image')

        self.imgList = os.listdir(self.imagePath)

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, index):
        imgName = self.imgList[index]
        img = cv2.imread(os.path.join(self.imagePath, imgName))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        img = {'img': img}
        if self.transform:
            img = self.transform(img)
        return img

    def label2index(self, label):
        return self.labelDic['label2index'][label]

    def index2label(self, index):
        return self.labelDic['index2label'][index]

    def getImageName(self, index):
        imgName = self.imgList[index]
        return imgName


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}

def collater_test(data):
    imgs = [s['img'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, common_size=512):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}

class Resizer_test(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, common_size=512):
        image = sample['img']
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        return {'img': torch.from_numpy(new_image), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.25081185, 0.57518238, 0.33086172]]])
        self.std = np.array([[[0.05553823, 0.11158981, 0.07569035]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}

class Normalizer_test(object):

    def __init__(self):
        self.mean = np.array([[[0.25081185, 0.57518238, 0.33086172]]])
        self.std = np.array([[[0.05553823, 0.11158981, 0.07569035]]])

    def __call__(self, sample):
        image = sample['img']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std)}


if __name__ == "__main__":
    dataset = TrainDataset()
    mean = np.zeros(3)
    std = np.zeros(3)
    for d in tqdm(dataset):
        img = d['img']
        for i in range(3):
            mean[i] += img[:, :, i].mean()
            std[i] += img[:, :, i].std()
    mean = mean / len(dataset)
    std = std / len(dataset)
    print(mean, std)