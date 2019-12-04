import os
from tqdm import tqdm
import pickle
import random
import numbers
import itertools
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

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
        for i in tqdm(range(len(self)),desc='Creating Photo Dataset'):
            frame = self[i]['frame']
            ann = self[i]['annotations']
            photo_file_name = os.path.join(photos_dir,'%d.png'%i)
            cv2.imwrite(photo_file_name,frame)
            for k,v in ann['template_matched'].items(): # Take first available annotation for now
                if v is None:
                    continue
                annotations[i] = v
                break
            if i not in annotations:
                annotations[i] = None
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

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.resnet18(pretrained=True)
        self.seq = torch.nn.Sequential(
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4,
            self.base.avgpool
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=256),
            torch.nn.Linear(in_features=256,out_features=1)
        )
        self.coordinate = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=256),
            torch.nn.Linear(in_features=256,out_features=2),
            torch.nn.Sigmoid()
        )
    def forward(self,x):
        x = self.seq(x)
        x = x.squeeze()
        visible = self.classifier(x)
        coord = self.coordinate(x)
        return coord, visible

def output_predictions(file_name,x,vis_pred,coord_pred,n=5):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    unnormalize = torchvision.transforms.Normalize(
            mean=-mean/std,
            std=1/std,
            inplace=False
    )

    output = []
    batch_size = min(x['image'].shape[0],n)
    for i in range(batch_size):
        img = unnormalize(x['image'][i]).permute(1,2,0).numpy()*255
        img = img.copy()
        w,h,_ = img.shape
        # Draw ground truth
        if x['visible'][i] > 0.5:
            cx,cy = (x['coordinates'][i]*torch.tensor([w,h])).long()
            cv2.line(img,(cx-5,cy),(cx+5,cy),(255,0,0))
            cv2.line(img,(cx,cy-5),(cx,cy+5),(255,0,0))
        # Draw prediction
        cx,cy = (coord_pred[i]*torch.tensor([w,h])).long()
        if vis_pred[i] > 0:
            cv2.line(img,(cx-5,cy),(cx+5,cy),(0,255,0))
            cv2.line(img,(cx,cy-5),(cx,cy+5),(0,255,0))
        cv2.putText(img, '%.2f'%torch.sigmoid(vis_pred[i]), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
        # Add to output
        output.append(img)
    # Concatenate outputs
    output = np.concatenate(output,axis=1)
    # Save image
    cv2.imwrite(file_name,output)

if __name__=='__main__':
    #d = VideoDataset('/home/howard/Code/video-annotator/smalldataset')
    #d = VideoDataset('/home/howard/Code/video-annotator/dataset')
    #d.to_photo_dataset()

    train_transform = torchvision.transforms.Compose([
        RandomCrop(224),
        #CentreCrop(500),
        ToTensor(),
        Normalize(),
    ])
    test_transform = torchvision.transforms.Compose([
        CentreCrop(224),
        ToTensor(),
        Normalize(),
    ])
    train_dataset = PhotoDataset('/home/howard/Code/video-annotator/smalldataset', transform=train_transform)
    test_dataset = PhotoDataset('/home/howard/Code/video-annotator/smalldataset', transform=test_transform)
    #test_dataset = PhotoDataset('/home/howard/Code/video-annotator/dataset', transform=test_transform)
    #train_dataset = torch.utils.data.Subset(train_dataset,range(3))
    test_dataset = torch.utils.data.Subset(test_dataset,range(3))
    #train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=torch.utils.data.RandomSampler(train_dataset,replacement=True,num_samples=16))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    net = Net()
    for p in net.seq.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    vis_criterion = torch.nn.BCEWithLogitsLoss()
    coord_criterion = torch.nn.MSELoss(reduce=False)
    train_loss_history = []
    test_loss_history = []
    tqdm = lambda x: x
    for _ in itertools.count():
        test_total_loss = 0
        test_total_vis_loss = 0
        test_total_coord_loss = 0
        net.eval()
        for x in tqdm(test_dataloader):
            vis = x['visible'].float().view(-1,1)
            coord = x['coordinates']
            coord_pred,vis_pred = net(x['image'])
            vis_pred = vis_pred.view(-1,1)
            vis_loss = vis_criterion(vis_pred,vis)
            coord_loss = coord_criterion(coord_pred,coord)
            coord_loss = (coord_loss*vis).sum()
            loss = vis_loss + coord_loss

            test_total_loss += loss.item()
            test_total_vis_loss += vis_loss.item()
            test_total_coord_loss += coord_loss.item()
        output_predictions('test_predictions.png',x,vis_pred,coord_pred)

        total_loss = 0
        total_vis_loss = 0
        total_coord_loss = 0
        net.train()
        for x in tqdm(train_dataloader):
            vis = x['visible'].float().view(-1,1)
            coord = x['coordinates']
            coord_pred,vis_pred = net(x['image'])
            vis_pred = vis_pred.view(-1,1)
            vis_loss = vis_criterion(vis_pred,vis)
            coord_loss = coord_criterion(coord_pred,coord)
            coord_loss = (coord_loss*vis).sum()
            loss = vis_loss + coord_loss

            total_loss += loss.item()
            total_vis_loss += vis_loss.item()
            total_coord_loss += coord_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        output_predictions('train_predictions.png',x,vis_pred,coord_pred)
        print('Test',test_total_loss, test_total_vis_loss, test_total_coord_loss)
        print('Train',total_loss, total_vis_loss, total_coord_loss)
        train_loss_history.append(total_loss)
        test_loss_history.append(test_total_loss)

        plt.plot(range(len(train_loss_history)), train_loss_history,label='Training Loss')
        plt.plot(range(len(test_loss_history)), test_loss_history,label='Testing Loss')
        plt.grid()
        plt.legend(loc='best')
        plt.savefig('plot.png')
        plt.close()