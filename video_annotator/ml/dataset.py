import os
from tqdm import tqdm
import pickle
import random
import numbers

import cv2
import torch
import torchvision

import video_annotator
from video_annotator.annotation import Annotations
from video_annotator.video import Video

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.videos_dir = os.path.join(dataset_dir,'videos')
        self.annotations_dir = os.path.join(dataset_dir,'annotations')

        self.file_names = [f for f in os.listdir(self.videos_dir) if os.path.isfile(os.path.join(self.videos_dir, f))]
        self.annotations = {}
        self.videos = {}

        self.dataset_index_mapping = []
        for vid_index in range(len(self.file_names)):
            vid = self.get_video(vid_index)
            fc = vid.frame_count
            for frame_index in range(0,fc,fc//10):
                self.dataset_index_mapping.append([vid_index,frame_index])
        for i in range(len(self.file_names)):
            for ann in self.get_annotations(i).annotations.values():
                ann.template_matched.generate(0)

    def get_video(self,index):
        if index not in self.videos:
            video_file_name = self.file_names[index]
            video_file_path = os.path.join(self.videos_dir,video_file_name)
            self.videos[index] = Video(video_file_path)
        return self.videos[index]

    def get_annotations(self,index):
        if index not in self.annotations:
            video_file_name = self.file_names[index]
            ann_file_name = video_file_name.split('.')[0]+'.pkl'
            ann_file_path = os.path.join(self.annotations_dir,ann_file_name)
            self.annotations[index] = Annotations(ann_file_path,self.get_video(index))
        return self.annotations[index]

    def __getitem__(self,index):
        vid_index,frame_index = self.dataset_index_mapping[index]
        return {
                'frame': self.get_video(vid_index)[frame_index],
                'annotations': self.get_annotations(vid_index).slice(frame_index)
        }

    def __len__(self):
        return len(self.dataset_index_mapping)

    def to_photo_dataset(self):
        photos_dir = os.path.join(self.dataset_dir,'photos')
        if not os.path.isdir(photos_dir):
            os.makedirs(photos_dir)
        annotations = {}
        for i in tqdm(range(len(self))):
            frame = self[i]['frame']
            ann = self[i]['annotations']
            photo_file_name = os.path.join(photos_dir,'%d.png'%i)
            cv2.imwrite(photo_file_name,frame)
            for k,v in ann['template_matched'].items(): # Take first available annotation for now
                if v is None:
                    continue
                annotations[i] = v
                break
        with open(os.path.join(self.dataset_dir,'photo-annotations.pkl'), 'wb') as f:
            pickle.dump(annotations, f)

class PhotoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.photos_dir = os.path.join(dataset_dir,'photos')
        self.annotations_path = os.path.join(dataset_dir,'photo-annotations.pkl')

        self.transform = transform

        with open(self.annotations_path,'rb') as f:
            self.annotations = pickle.load(f)

    def __getitem__(self,index):
        photo_file_name = os.path.join(self.photos_dir,'%d.png'%index)
        output = {
                'image': cv2.imread(photo_file_name),
                'coordinates': self.annotations[index],
		'visible': self.annotations[index] is not None
        }
        if self.transform is not None:
            output = self.transform(output)
        return output

    def __len__(self):
        return len(self.annotations.keys())

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

        i,j,th,tw = self.get_params(output['image'],self.size)
        output['image'] = self.crop_image((i,j), self.size, sample['image'])
        output['coordinates'] = self.crop_coordinates((i,j), self.size, sample['image'], sample['coordinates'])
        output['visible'] = self.crop_visible(sample['visible'],output['coordinates'])
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

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = torchvision.transforms.Normalize(
                mean=mean,
                std=std,
                inplace=False
        )
    def __call__(self, sample):
        output = sample.copy()
        output['image'] = self.transform(sample['image'])
        return output

class ToTensor(object):
    def __init__(self):
        self.transform = torchvision.transforms.ToTensor()
    def __call__(self, sample):
        output = sample.copy()
        output['image'] = self.transform(sample['image'])
        if sample['coordinates'] is None:
            output['coordinates'] = torch.empty([2])
        else:
            output['coordinates'] = torch.tensor(sample['coordinates'])
        output['visible'] = torch.tensor(sample['visible'])
        return output

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=6,kernel_size=(6,6),stride=2),
            torch.nn.Conv2d(in_channels=6,out_channels=12,kernel_size=(6,6),stride=2),
            torch.nn.Conv2d(in_channels=12,out_channels=24,kernel_size=(6,6),stride=2),
            torch.nn.Conv2d(in_channels=24,out_channels=48,kernel_size=(6,6),stride=2),
            torch.nn.Conv2d(in_channels=48,out_channels=96,kernel_size=(6,6),stride=2),
            torch.nn.Conv2d(in_channels=96,out_channels=96,kernel_size=(11,11))
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=96,out_features=48),
            torch.nn.Linear(in_features=48,out_features=1)
        )
        self.coordinate = torch.nn.Sequential(
            torch.nn.Linear(in_features=96,out_features=48),
            torch.nn.Linear(in_features=48,out_features=2)
        )
    def forward(self,x):
        x = self.seq(x)
        x = x.squeeze()
        visible = self.classifier(x)
        coord = self.coordinate(x)
        return coord, visible

if __name__=='__main__':
    #d = VideoDataset('/home/howard/Code/video-annotator/smalldataset')
    #d = VideoDataset('/home/howard/Code/video-annotator/dataset')
    #d.to_photo_dataset()

    train_transform = torchvision.transforms.Compose([
        RandomCrop(500),
        ToTensor()
    ])
    test_transform = torchvision.transforms.Compose([
        CentreCrop(500),
        ToTensor()
    ])
    train_dataset = PhotoDataset('/home/howard/Code/video-annotator/smalldataset', transform=train_transform)
    test_dataset = PhotoDataset('/home/howard/Code/video-annotator/smalldataset', transform=test_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    net = Net()

    optimizer = torch.optim.Adam(net.parameters())

    vis_criterion = torch.nn.BCEWithLogitsLoss()
    coord_criterion = torch.nn.MSELoss()
    #while True:
    for _ in range(10):
        test_total_loss = 0
        test_total_vis_loss = 0
        test_total_coord_loss = 0
        for x in tqdm(test_dataloader):
            vis = x['visible'].float().squeeze()
            coord = x['coordinates']
            coord_pred,vis_pred = net(x['image'])
            vis_pred = vis_pred.squeeze()
            vis_loss = vis_criterion(vis_pred,vis)
            coord_loss = coord_criterion(coord_pred,coord)
            loss = vis_loss + coord_loss

            test_total_loss += loss.item()
            test_total_vis_loss += vis_loss.item()
            test_total_coord_loss += coord_loss.item()

        total_loss = 0
        total_vis_loss = 0
        total_coord_loss = 0
        for x in tqdm(train_dataloader):
            vis = x['visible'].float().squeeze()
            coord = x['coordinates']
            coord_pred,vis_pred = net(x['image'])
            vis_pred = vis_pred.squeeze()
            vis_loss = vis_criterion(vis_pred,vis)
            coord_loss = coord_criterion(coord_pred,coord)
            loss = vis_loss + coord_loss

            total_loss += loss.item()
            total_vis_loss += vis_loss.item()
            total_coord_loss += coord_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Test',test_total_loss, test_total_vis_loss, test_total_coord_loss)
        print('Train',total_loss, total_vis_loss, total_coord_loss)
