import numpy as np
import random
import numbers
import cv2
import torch
import torchvision

class Scale(object):
    def __init__(self, size):
        self.size = size
    @staticmethod
    def get_params(img, size):
        h,w,_ = img.shape
        scale = max(size/h,size/w)
        return int(w*scale),int(h*scale)
    @staticmethod
    def scale_image(size, img):
        return cv2.resize(img,size)
    def __call__(self, sample):
        output = sample.copy()
        w,h = self.get_params(output['image'],self.size)
        output['image'] = cv2.resize(sample['image'],(w,h))
        return output

class RandomScale(object):
    def __init__(self, min_size):
        self.min_size = min_size
    @staticmethod
    def get_params(img, min_size):
        h,w,_ = img.shape
        mh = min_size
        mw = min_size
        min_scale = max(mh/h,mw/w)
        scale = random.random()*(1-min_scale)+min_scale
        return int(w*scale),int(h*scale)
    @staticmethod
    def scale_image(size, img):
        return img.resize(size)
    def __call__(self, sample):
        output = sample.copy()
        w,h = self.get_params(output['image'],self.min_size)
        output['image'] = cv2.resize(sample['image'],(w,h))
        return output

class RandomCrop(object):
    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    @staticmethod
    def get_params(img, size):
        h,w,_ = img.shape
        th, tw = size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
    @staticmethod
    def crop_image(topleft, size, img):
        i,j = topleft
        th, tw = size
        return img[i:i+th,j:j+tw,:]
    @staticmethod
    def crop_coordinates(topleft, size, img, coord):
        if coord is None:
            return None
        h,w,_ = img.shape
        i,j = topleft
        th, tw = size
        x,y = coord
        return (
            (w*x-j)/tw,
            (h*y-i)/th
        )
    @staticmethod
    def crop_visible(visible,annotation):
        if not visible:
            return False
        x,y = annotation
        if x < 0 or x >= 1:
            return False
        if y < 0 or y >= 1:
            return False
        return True
    def __call__(self, sample):
        output = sample.copy()

        i,j,_,_ = self.get_params(output['image'],self.size)
        output['image'] = self.crop_image((i,j), self.size, sample['image'])
        output['coordinates'] = [self.crop_coordinates((i,j), self.size, sample['image'], c) for c in sample['coordinates']]
        output['visible'] = any([self.crop_visible(sample['visible'],c) for c in output['coordinates']])
        return output

class CentreCrop(RandomCrop):
    @staticmethod
    def get_params(img, size):
        h,w,_ = img.shape
        th, tw = size
        if w == tw and h == th:
            return 0, 0, h, w
        i = (h - th)//2
        j = (w - tw)//2
        return i, j, th, tw

class RandomHorizontalFlip(RandomCrop):
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        if np.random.rand() > self.prob:
            return sample

        output = sample.copy()
        output['image'] = np.fliplr(sample['image']).copy()
        output['coordinates'] = [(1-c[0],c[1]) for c in sample['coordinates']]
        return output

class Normalize(object):
    def __init__(self, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
        self.transform = torchvision.transforms.Normalize(
                mean=mean,
                std=std,
                inplace=False
        )
    def __call__(self, sample):
        output = sample.copy()
        output['image'] = self.transform(sample['image'])
        return output

class FilterCoords(object):
    def __call__(self, sample):
        coord = None
        for c in sample['coordinates']:
            # Check of annotation exists
            if c is None:
                continue
            # Check if out of bounds
            if c[0] < 0 or c[0] > 1 or c[1] < 0 or c[1] > 1:
                continue
            # Return first keypoint that is visible
            coord = c
            break
        output = sample.copy()
        output['coordinates'] = coord
        output['visible'] = coord is not None
        return output

class ToTensor(object):
    def __init__(self):
        self.transform = torchvision.transforms.ToTensor()
    def __call__(self, sample):
        output = sample.copy()
        output['image'] = self.transform(sample['image'])
        if sample['coordinates'] is None:
            output['coordinates'] = torch.zeros([2])
        else:
            output['coordinates'] = torch.tensor(sample['coordinates'])
        output['visible'] = torch.tensor(sample['visible'])
        return output

class ToTensorAnchorBox(object):
    """ Convert to Tensors with coordinates in anchor box format. """
    def __init__(self, divs=[7,7]):
        self.transform = torchvision.transforms.ToTensor()
        self.divs = divs
    def __call__(self, sample):
        coords = torch.zeros([2]+self.divs)
        vis = torch.zeros([1]+self.divs)
        d = torch.tensor(self.divs)
        for c in sample['coordinates']:
            # Check if visible
            if c is None:
                continue
            if c[0] < 0 or c[0] > 1 or c[1] < 0 or c[1] > 1:
                continue
            # Compute which anchor box it belongs to
            c = torch.tensor(c)
            discrete_coord = torch.min((c*d).long(),d-1) # If c==1, then it goes out of bounds. Use max to fix that.
            # Check if another keypoint was already found in this anchor box
            if vis[0,discrete_coord[0],discrete_coord[1]]:
                continue
            vis[0,discrete_coord[0],discrete_coord[1]] = 1
            # Compute coordinate relative to top-left of anchor box
            coords[:,discrete_coord[0],discrete_coord[1]] = c*d-discrete_coord

        output = sample.copy()
        output['image'] = self.transform(sample['image'])
        output['coordinates'] = coords
        output['visible'] = vis
        return output
