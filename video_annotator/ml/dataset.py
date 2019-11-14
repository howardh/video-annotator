import os
from tqdm import tqdm
import pickle

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
        self.file_names = self.file_names[:1]
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
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.photos_dir = os.path.join(dataset_dir,'photos')
        self.annotations_path = os.path.join(dataset_dir,'photo-annotations.pkl')

        with open(self.annotations_path,'rb') as f:
            self.annotations = pickle.load(f)

    def __getitem__(self,index):
        photo_file_name = os.path.join(self.photos_dir,'%d.png'%index)
        return {
                'photo': cv2.imread(photo_file_name),
                'annotation': self.annotations[index]
        }

    def __len__(self):
        return len(self.annotations.keys())

if __name__=='__main__':
    d = VideoDataset('/home/howard/Code/video-annotator/smalldataset')
    d.to_photo_dataset()
    d = PhotoDataset('/home/howard/Code/video-annotator/smalldataset')
