import cv2
import time
import pickle
import os
from tqdm import tqdm
from collections import defaultdict

import templatematcher

def interpolate_annotations(points):
    if len(points) == 0:
        return []
    keyframes = sorted(points.keys())
    num_frames = keyframes[-1]+1
    output = [None]*num_frames
    for start,end in zip(keyframes,keyframes[1:]):
        if points[start] is None or points[end] is None:
            continue
        diff = (points[end][0]-points[start][0],points[end][1]-points[start][1])
        diff_per_frame = (diff[0]/(end-start),diff[1]/(end-start))
        for i in range(end-start+1):
            output[start+i] = (points[start][0]+diff_per_frame[0]*i,points[start][1]+diff_per_frame[1]*i)
    return output

class Annotations():
    def __init__(self, file_path):
        self.file_path = file_path
        self.annotations = {}
        self.interpolated_annotations = {}
        self.generated_annotations = defaultdict(lambda: [])

        self.load_annotations(file_path)
        self.interpolate_annotations()

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
        self.interpolated_annotations[annotation_id] = interpolate_annotations(self.annotations[annotation_id])

    def remove_annotation(self, frame_index, annotation_id):
        if frame_index not in self.annotations[annotation_id]:
            print('No keyframe selected')
            return
        del self.annotations[annotation_id][frame_index]
        self.interpolated_annotations[annotation_id] = interpolate_annotations(self.annotations[annotation_id])

    def get_annotation(self, frame_index, annotation_id):
        return self.interpolated_annotations[annotation_id][frame_index]

    def load_annotations(self, annotation_file_path):
        if os.path.isfile(annotation_file_path):
            with open(annotation_file_path, 'rb') as f:
                self.annotations = pickle.load(f)
        else:
            self.annotations = {}

    def interpolate_annotations(self):
        for k,v in self.annotations.items():
            self.interpolated_annotations[k] = interpolate_annotations(v)
    
    def generate_annotations(self, annotation_id, video):
        num_frames = max(self.annotations[annotation_id])
        annotations = [None]*num_frames
        self.generated_annotations[annotation_id] = annotations
        try:
            for frame_index in tqdm(range(num_frames),desc='Generating annotations'):
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
            interp_ann = self.interpolated_annotations[ann_id]
            gen_ann = self.generated_annotations[ann_id]
            if frame_index < len(interp_ann) and interp_ann[frame_index] is not None:
                centre = interp_ann[frame_index]
                centre = (int(centre[0]*width),
                          int(centre[1]*height))
                cv2.circle(frame, center=centre,
                        radius=10, color=(0,255,0),
                        thickness=5, lineType=8, shift=0)
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
        if index > len(self.data):
            self.data = self.data + [None]*(index-len(self.data)+1)
        self.data[index] = value
