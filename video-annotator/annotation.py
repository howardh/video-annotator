import cv2
import time
import pickle
import os
from tqdm import tqdm
from collections import defaultdict

import templatematcher

class Annotations():
    def __init__(self, file_path, video):
        self.file_path = file_path
        self.video = video
        self.annotations = defaultdict(lambda: SparseAnnotation())
        self.generated_annotations = defaultdict(lambda: DenseAnnotation())

        self.load_annotations(file_path)

    def __getitem__(self, annotation_id):
        if annotation_id not in self.annotations:
            self.annotations[annotation_id] = {}
        return self.annotations[annotation_id]

    def __delitem__(self, annotation_id):
        del self.annotations[annotation_id]

    def get_ids(self):
        return self.annotations.keys()

    def add_annotation(self, frame_index, annotation_id, annotation):
        if annotation_id not in self.annotations:
            self.annotations[annotation_id] = {}
        self.annotations[annotation_id][frame_index] = annotation

    def remove_annotation(self, frame_index, annotation_id):
        if frame_index not in self.annotations[annotation_id]:
            print('No keyframe selected')
            return
        del self.annotations[annotation_id][frame_index]

    def load_annotations(self, annotation_file_path):
        if os.path.isfile(annotation_file_path):
            with open(annotation_file_path, 'rb') as f:
                self.annotations = pickle.load(f)
        else:
            self.annotations = {}

    def generate_annotations(self, annotation_id, starting_index=0):
        num_frames = self.video.frame_count
        annotations = self.generated_annotations[annotation_id]
        try:
            for frame_index in tqdm(range(starting_index,num_frames),desc='Generating annotations'):
                ann = templatematcher.generate_annotation(video,self,
                        frame_index,
                        annotation_id=annotation_id,
                        window_size=(128,128))
                annotations[frame_index] = ann[annotation_id]
        except KeyboardInterrupt:
            pass
        return annotations

    def save_annotations(self, annotation_file_path):
        with open(annotation_file_path, 'wb') as f:
            pickle.dump(self.annotations, f)

    def render(self, frame, frame_index, num_frames=100):
        height,width,_ = frame.shape
        for ann_id in self.annotations.keys():
            manu_ann = self.annotations[ann_id]
            if frame_index in manu_ann and manu_ann[frame_index] is not None:
                centre = manu_ann[frame_index]
                centre = (int(centre[0]*width),
                          int(centre[1]*height))
                cv2.circle(frame, center=centre,
                        radius=10, color=(0,255,0),
                        thickness=5, lineType=8, shift=0)
            if ann_id in self.generated_annotations:
                gen_ann = self.generated_annotations[ann_id]
                if frame_index < len(gen_ann) and gen_ann[frame_index] is not None:
                    centre = gen_ann[frame_index]
                    centre = (int(centre[0]*width),
                              int(centre[1]*height))
                    cv2.circle(frame, center=centre,
                            radius=10, color=(255,0,0),
                            thickness=5, lineType=8, shift=0)
                if len(gen_ann) > 0:
                    for i in range(max(1,frame_index-num_frames),frame_index):
                        c0 = gen_ann[i-1]
                        c1 = gen_ann[i]
                        if c0 is None or c1 is None:
                            continue
                        c0 = (int(c0[0]*width),int(c0[1]*height))
                        c1 = (int(c1[0]*width),int(c1[1]*height))
                        cv2.line(frame,c0,c1,color=(255,0,0),thickness=3)
        return frame

class Annotation():
    def __init__(self):
        pass

class SparseAnnotation():
    def __init__(self):
        self.data = {}

    def __getitem__(self,index):
        if index in self.data:
            return self.data[index]
        else:
            return None

    def __setitem__(self,index,value):
        self.data[index] = value

class DenseAnnotation():
    def __init__(self):
        self.data = []

    def __getitem__(self,index):
        if index < len(self.data):
            return self.data[index]
        else:
            return None

    def __setitem__(self,index,value):
        if index >= len(self.data):
            self.data = self.data + [None]*(index-len(self.data)+1)
        self.data[index] = value

    def __len__(self):
        return len(self.data)
