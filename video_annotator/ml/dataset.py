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

from video_annotator.ml.model import Net, Net2
from video_annotator.ml.transforms import Scale, RandomScale, RandomCrop, CentreCrop, RandomHorizontalFlip, Normalize, FilterCoords, ToTensor, ToTensorAnchorBox

# Datasets

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, frames_per_vid=10):
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
            for frame_index in range(0,fc,fc//frames_per_vid):
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

    def to_photo_dataset(self, size=None):
        photos_dir = os.path.join(self.dataset_dir,'photos')
        if not os.path.isdir(photos_dir):
            os.makedirs(photos_dir)
        annotations = []
        for i in tqdm(range(len(self)),desc='Creating Photo Dataset'):
            frame = self[i]['frame']
            if size is not None:
                height, width, _ = frame.shape
                scale = size/min(width,height)
                frame = cv2.resize(frame,(int(width*scale),int(height*scale)))
            ann = self[i]['annotations']
            photo_file_name = os.path.join(photos_dir,'%d.png'%i)
            cv2.imwrite(photo_file_name,frame)
            annotations.append([])
            for k,v in ann['template_matched'].items():
                if v is None:
                    continue
                annotations[i].append(v)
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
		'visible': len(self.annotations[index]) > 0
        }
        if self.transform is not None:
            output = self.transform(output)
        return output

    def __len__(self):
        return len(self.annotations)

# Utils

def output_predictions(file_name,x,vis_pred,coord_pred,n=5):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    unnormalize = torchvision.transforms.Normalize(
            mean=-mean/std,
            std=1/std,
            inplace=False
    )

    output = []
    indices = [int((x['image'].shape[0]-1)/n*i) for i in range(n)]
    for i in indices:
        img = unnormalize(x['image'][i]).permute(1,2,0).numpy()*255
        img = img.copy()
        w,h,_ = img.shape
        # Draw ground truth
        if 'visible' in x and x['visible'][i] > 0.5:
            cx,cy = (x['coordinates'][i].cpu()*torch.tensor([w,h])).long()
            cv2.line(img,(cx-5,cy),(cx+5,cy),(255,0,0))
            cv2.line(img,(cx,cy-5),(cx,cy+5),(255,0,0))
        # Draw prediction
        cx,cy = (coord_pred[i].cpu()*torch.tensor([w,h])).long()
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

def output_predictions_anchor_box(file_name,x,vis_pred,coord_pred,n=5):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    unnormalize = torchvision.transforms.Normalize(
            mean=-mean/std,
            std=1/std,
            inplace=False
    )

    output = []
    indices = [int((x['image'].shape[0]-1)/n*i) for i in range(n)]
    boxes = torch.tensor(coord_pred.shape[-2:]) # Number of anchor boxes in each dimension
    def render_keypoints(img, vis, coord, colour=(0,0,0), render_confidence=True):
        w,h,_ = img.shape
        box_dims = 1/boxes.float()
        for box in itertools.product(range(boxes[0]),range(boxes[1])):
            # Anchor box indices
            box = torch.tensor(box)
            # Visibility
            v = vis[0,box[0],box[1]]
            if v < 0.5:
                continue
            # Coordinate relative to the upper-left corner of anchor box
            rel_coord = coord[:,box[0],box[1]]
            # Absolute coordinate in [0,1]^2
            abs_coord = (box+rel_coord)*box_dims
            # Compute coordinate in pixels
            cx,cy = (abs_coord*torch.tensor([w,h])).long()
            # Draw
            cv2.line(img,(cx-5,cy),(cx+5,cy),colour)
            cv2.line(img,(cx,cy-5),(cx,cy+5),colour)
            # Confidence
            if render_confidence:
                cv2.putText(img, '%.2f'%torch.sigmoid(v), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour)
        return img
    for i in indices:
        img = unnormalize(x['image'][i]).permute(1,2,0).numpy()*255
        img = img.copy()
        # Draw predictions
        img = render_keypoints(img, vis_pred[i], coord_pred[i], (0,255,0), render_confidence=True)
        # Draw ground truth
        img = render_keypoints(img, x['visible'][i], x['coordinates'][i], (255,0,0), render_confidence=False)
        # Add to output
        output.append(img)
    # Concatenate outputs
    output = np.concatenate(output,axis=1)
    # Save image
    cv2.imwrite(file_name,output)

####################################################################################################

def train():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('GPU found')
        device = torch.device('cuda')
    else:
        print('No GPU found')
        device = torch.device('cpu')

    train_transform = torchvision.transforms.Compose([
        Scale(300),
        RandomScale(224),
        RandomCrop(224),
        RandomHorizontalFlip(0.5),
        FilterCoords(),
        ToTensor(),
        Normalize(),
    ])
    test_transform = torchvision.transforms.Compose([
        Scale(int((300+224)/2)),
        CentreCrop(224),
        FilterCoords(),
        ToTensor(),
        Normalize(),
    ])

    # Overfit datasets (No stochasticity)
    train_dataset = PhotoDataset('dataset', transform=train_transform)
    test_dataset = PhotoDataset('smalldataset', transform=test_transform)

    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
            batch_size=121, shuffle=True, drop_last=True, pin_memory=use_gpu)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=44,
            shuffle=True, pin_memory=use_gpu)

    print('Train set size',len(train_dataset))
    print('Test set size', len(test_dataset))

    net = Net()
    #for p in net.seq.parameters():
    #    p.requires_grad = False
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    vis_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    coord_criterion = torch.nn.MSELoss(reduce=False)
    train_loss_history = []
    test_loss_history = []

    vw=1
    cw=1

    #tqdm = lambda x: x
    for iteration in itertools.count():
        # Save model
        if iteration % 100:
            with open('checkpoints/checkpoint-%d.pt'%iteration,'wb') as f:
                torch.save({
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_loss_history': train_loss_history,
                    'test_loss_history': test_loss_history,
                    'iteration': iteration
                },f)
            print('Checkpoint saved')

        # Test
        test_total_loss = 0
        test_total_vis_loss = 0
        test_total_coord_loss = 0
        net.eval()
        for x in tqdm(test_dataloader):
            vis = x['visible'].float().view(-1,1).to(device)
            coord = x['coordinates'].to(device)
            coord_pred,vis_pred = net(x['image'].to(device))
            vis_pred = vis_pred.view(-1,1)
            vis_loss = vis_criterion(vis_pred,vis).sum()
            vis_loss = vw*vis_loss
            coord_loss = coord_criterion(coord_pred,coord)
            coord_loss = (coord_loss*vis).sum()
            coord_loss = cw*coord_loss
            loss = vis_loss + coord_loss

            test_total_loss += loss.item()/len(test_dataset)
            test_total_vis_loss += vis_loss.item()/len(test_dataset)
            test_total_coord_loss += coord_loss.item()/len(test_dataset)
        output_predictions('figs/test_predictions.png',x,vis_pred,coord_pred)

        # Train
        total_loss = 0
        total_vis_loss = 0
        total_coord_loss = 0
        net.train()
        visible_count = 0
        batch_count = 0
        for x in tqdm(train_dataloader):
            vis = x['visible'].float().view(-1,1).to(device)
            coord = x['coordinates'].to(device)
            coord_pred,vis_pred = net(x['image'].to(device))
            vis_pred = vis_pred.view(-1,1)
            num_samples = vis.shape[0]
            num_vis = max(vis.sum(),1)
            num_invis = max(num_samples-num_vis,1)
            vis_weight = vis*(1/num_vis) + (1-vis)*(1/num_invis)
            vis_loss = (vis_weight*vis_criterion(vis_pred,vis)).sum()
            vis_loss = vw*vis_loss
            coord_loss = coord_criterion(coord_pred,coord)
            coord_loss = (coord_loss*vis).sum()
            coord_loss = cw*coord_loss
            loss = vis_loss + coord_loss

            total_loss += loss.item()/len(train_dataset)
            total_vis_loss += vis_loss.item()/len(train_dataset)
            total_coord_loss += coord_loss.item()/len(train_dataset)
            visible_count += vis.sum()
            batch_count += vis.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        output_predictions('figs/train_predictions.png',x,vis_pred,coord_pred)
        print('Iteration',iteration)
        print('Test',test_total_loss, test_total_vis_loss, test_total_coord_loss)
        print('Train',total_loss, total_vis_loss, total_coord_loss)
        print('Vis',visible_count/batch_count)
        train_loss_history.append(total_loss)
        test_loss_history.append(test_total_loss)

        plt.plot(range(len(train_loss_history)), train_loss_history,label='Training Loss')
        plt.plot(range(len(test_loss_history)), test_loss_history,label='Testing Loss')
        plt.xlabel('# Iterations')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend(loc='best')
        plt.savefig('figs/plot.png')
        plt.close()

        plt.plot(range(len(train_loss_history)), [np.log(x) for x in train_loss_history],label='Training Loss')
        plt.plot(range(len(test_loss_history)), [np.log(x) for x in test_loss_history],label='Testing Loss')
        plt.xlabel('# Iterations')
        plt.ylabel('Log Loss')
        plt.grid()
        plt.legend(loc='best')
        plt.savefig('figs/log-plot.png')
        plt.close()

def train_anchor_box():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('GPU found')
        device = torch.device('cuda')
    else:
        print('No GPU found')
        device = torch.device('cpu')

    train_transform = torchvision.transforms.Compose([
        Scale(300),
        RandomScale(224),
        RandomCrop(224),
        RandomHorizontalFlip(0.5),
        ToTensorAnchorBox(),
        Normalize(),
    ])
    test_transform = torchvision.transforms.Compose([
        Scale(int((300+224)/2)),
        CentreCrop(224),
        ToTensorAnchorBox(),
        Normalize(),
    ])

    # Overfit datasets (No stochasticity)
    train_dataset = PhotoDataset('dataset', transform=train_transform)
    test_dataset = PhotoDataset('smalldataset', transform=test_transform)

    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
            batch_size=121, shuffle=True, drop_last=True, pin_memory=use_gpu)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=44,
            shuffle=True, pin_memory=use_gpu)

    print('Train set size',len(train_dataset))
    print('Test set size', len(test_dataset))

    net = Net2()
    #for p in net.seq.parameters():
    #    p.requires_grad = False
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    vis_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    coord_criterion = torch.nn.MSELoss(reduce=False)
    train_loss_history = []
    test_loss_history = []

    vw=1
    cw=1

    #tqdm = lambda x: x
    for iteration in itertools.count():
        # Save model
        if iteration % 100:
            with open('checkpoints/checkpoint2-%d.pt'%iteration,'wb') as f:
                torch.save({
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_loss_history': train_loss_history,
                    'test_loss_history': test_loss_history,
                    'iteration': iteration
                },f)
            print('Checkpoint saved')

        # Test
        test_total_loss = 0
        test_total_vis_loss = 0
        test_total_coord_loss = 0
        net.eval()
        for x in tqdm(test_dataloader):
            # Ground truth
            vis = x['visible'].float().to(device)
            coord = x['coordinates'].to(device)
            # Prediction
            coord_pred,vis_pred = net(x['image'].to(device))
            # Loss for each component
            vis_loss = vis_criterion(vis_pred,vis).sum()
            coord_loss = coord_criterion(coord_pred,coord)
            coord_loss = (coord_loss*vis).sum()
            # Weight losses
            vis_loss = vw*vis_loss
            coord_loss = cw*coord_loss
            # Overall loss
            loss = vis_loss + coord_loss
            # Sum losses
            test_total_loss += loss.item()/len(test_dataset)
            test_total_vis_loss += vis_loss.item()/len(test_dataset)
            test_total_coord_loss += coord_loss.item()/len(test_dataset)
        output_predictions_anchor_box('figs/test_predictions.png',x,vis_pred,coord_pred)

        # Train
        total_loss = 0
        total_vis_loss = 0
        total_coord_loss = 0
        net.train()
        visible_count = 0
        batch_count = 0
        for x in tqdm(train_dataloader):
            # Ground truth
            vis = x['visible'].float().to(device)
            coord = x['coordinates'].to(device)
            # Prediction
            coord_pred,vis_pred = net(x['image'].to(device))
            # Loss for each component
            vis_loss = vis_criterion(vis_pred,vis).sum()
            coord_loss = coord_criterion(coord_pred,coord)
            coord_loss = (coord_loss*vis).sum()
            # Weight losses
            vis_loss = vw*vis_loss
            coord_loss = cw*coord_loss
            # Overall loss
            loss = vis_loss + coord_loss
            # Sum losses
            total_loss += loss.item()/len(test_dataset)
            total_vis_loss += vis_loss.item()/len(test_dataset)
            total_coord_loss += coord_loss.item()/len(test_dataset)

            visible_count += vis.sum()
            batch_count += vis.shape[0]*vis.shape[1]*vis.shape[2]*vis.shape[3]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        output_predictions_anchor_box('figs/train_predictions.png',x,vis_pred,coord_pred)
        print('Iteration',iteration)
        print('Test',test_total_loss, test_total_vis_loss, test_total_coord_loss)
        print('Train',total_loss, total_vis_loss, total_coord_loss)
        print('Vis',visible_count/batch_count)
        train_loss_history.append(total_loss)
        test_loss_history.append(test_total_loss)

        plt.plot(range(len(train_loss_history)), train_loss_history,label='Training Loss')
        plt.plot(range(len(test_loss_history)), test_loss_history,label='Testing Loss')
        plt.xlabel('# Iterations')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend(loc='best')
        plt.savefig('figs/plot.png')
        plt.close()

        plt.plot(range(len(train_loss_history)), [np.log(x) for x in train_loss_history],label='Training Loss')
        plt.plot(range(len(test_loss_history)), [np.log(x) for x in test_loss_history],label='Testing Loss')
        plt.xlabel('# Iterations')
        plt.ylabel('Log Loss')
        plt.grid()
        plt.legend(loc='best')
        plt.savefig('figs/log-plot.png')
        plt.close()

def make_datasets():
    #d = VideoDataset('/home/howard/Code/video-annotator/smalldataset')
    d = VideoDataset('/home/howard/Code/video-annotator/dataset')
    d.to_photo_dataset(size=300)

if __name__=='__main__':
    #with open('checkpoint2.pt','rb') as f:
    #    checkpoint = torch.load(f,map_location=torch.device('cpu'))
    #net = Net()
    #net.load_state_dict(checkpoint['model'])
    train_anchor_box()
