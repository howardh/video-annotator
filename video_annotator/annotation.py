import cv2
import time
import pickle
import os
from tqdm import tqdm
from collections import defaultdict
import logging

import video_annotator
from video_annotator.templatematcher import Templates

log = logging.getLogger(__name__)

class Annotations():
    def __init__(self, file_path, video=None):
        self.file_path = file_path
        self.video = video
        self.annotations = {}

        self.load_annotations(file_path)

    def __getitem__(self, annotation_id):
        if annotation_id not in self.annotations:
            self.annotations[annotation_id] = Annotation(self.video)
        return self.annotations[annotation_id]

    def __delitem__(self, annotation_id):
        del self.annotations[annotation_id]

    def slice(self,frame_index):
        output = {
                'manual': {},
                'template_matched': {}
        }
        for ann_id in self.get_ids():
            ann = self.annotations[ann_id]
            output['manual'][ann_id] = ann.manual[frame_index]
            output['template_matched'][ann_id] = ann.template_matched[frame_index]
        return output

    def get_ids(self):
        return self.annotations.keys()

    def add_annotation(self, frame_index, annotation_id, annotation):
        self.annotations[annotation_id][frame_index] = annotation

    def remove_annotation(self, frame_index, annotation_id):
        del self.annotations[annotation_id][frame_index]

    def load_annotations(self, annotation_file_path):
        if os.path.isfile(annotation_file_path):
            with open(annotation_file_path, 'rb') as f:
                annotations = pickle.load(f)
            for ann_id,ann in annotations.items():
                self[ann_id].load(ann)
        else:
            self.annotations = {}

    def save_annotations(self, annotation_file_path=None):
        if annotation_file_path is None:
            annotation_file_path = self.file_path
        output = {}
        for ann_id,ann in self.annotations.items():
            output[ann_id] = {
                    'manual': ann.manual.data,
                    'template_matched': ann.template_matched.data
            }
        with open(annotation_file_path, 'wb') as f:
            pickle.dump(output, f)
        print('saving to', annotation_file_path)

    def render(self, frame, frame_index, num_frames=100):
        height,width,_ = frame.shape
        for ann_id in self.annotations.keys():
            ann = self[ann_id][frame_index]
            manu_ann = ann['manual']
            if manu_ann is not None:
                centre = manu_ann
                cx,cy = (int(centre[0]*width),
                        int(centre[1]*height))
                cs = 10 # Cross size
                cv2.line(frame,(cx-cs,cy),(cx+cs,cy),
                        color=(0,255,0),thickness=1)
                cv2.line(frame,(cx,cy-cs),(cx,cy+cs),
                        color=(0,255,0),thickness=1)
            gen_ann = ann['template_matched']
            if gen_ann is not None:
                centre = gen_ann
                cx,cy = (int(centre[0]*width),
                        int(centre[1]*height))
                cs = 10 # Cross size
                cv2.line(frame,(cx-cs,cy),(cx+cs,cy),
                        color=(255,0,0),thickness=1)
                cv2.line(frame,(cx,cy-cs),(cx,cy+cs),
                        color=(255,0,0),thickness=1)
            gen_path = self[ann_id].template_matched[frame_index-num_frames:frame_index]
            for c0,c1 in zip(gen_path,gen_path[1:]):
                if c0 is None or c1 is None:
                    continue
                c0 = (int(c0[0]*width),int(c0[1]*height))
                c1 = (int(c1[0]*width),int(c1[1]*height))
                cv2.line(frame,c0,c1,color=(255,0,0),thickness=2)
        return frame

class Annotation():
    def __init__(self, video):
        self.manual = SparseAnnotation()
        self.template_matched = TemplateMatchedAnnotations(video,self.manual)
    def __getitem__(self, frame_index):
        return {
            'manual': self.manual[frame_index],
            'template_matched': self.template_matched[frame_index]
        }
    def __setitem__(self, frame_index, annotation):
        self.manual[frame_index] = annotation
    def __delitem__(self, frame_index):
        del self.manual[frame_index]
    def load(self,data):
        self.manual.data = data['manual']
        self.template_matched.data = data['template_matched']

class SparseAnnotation():
    def __init__(self):
        self.data = {}
    def __getitem__(self,index):
        if index in self.data:
            return self.data[index]
        else:
            return None
    def __delitem__(self,index):
        del self.data[index]
    def __setitem__(self,index,value):
        self.data[index] = value
    def __len__(self):
        return len(self.data)
    def items(self):
        return self.data.items()
    def has_key(self, k):
        return k in self.data
    def update(self, *args, **kwargs):
        return self.data.update(*args, **kwargs)
    def keys(self):
        return self.data.keys()
    def values(self):
        return self.data.values()
    def nearest_indices(self,index):
        if index in self.data:
            return (index,index)
        keys = sorted(self.data.keys())
        low = keys[0]
        high = keys[-1]
        for k in keys:
            if k < index and k > low:
                low = k
            if k > index and k < high:
                high = k
                break
        return (low,high)

class DenseAnnotation():
    def __init__(self):
        self.data = []
    def __getitem__(self,index):
        if type(index) is slice:
            start,stop,step = index.start,index.stop,index.step
            if step is not None and step != 1:
                raise NotImplementedError(
                        'Slicing with >1 step size is not supported.')

            if start is None:
                start = 0
            if stop is None:
                stop = len(self.data)

            if start >= 0 and stop <= len(self.data):
                return self.data[start:stop]
            if start >= len(self.data) or stop <= 0:
                return [None]*(stop-start)
            if stop > len(self.data):
                return self[start:]+[None]*(stop-len(self.data))
            if start < 0:
                return [None]*(-start)+self[:stop]
        else:
            if index < len(self.data) and index >= 0:
                return self.data[index]
            else:
                return None
    def __setitem__(self,index,value):
        if index >= len(self.data):
            self.data = self.data + [None]*(index-len(self.data)+1)
        self.data[index] = value
    def __len__(self):
        return len(self.data)

class TemplateMatchedAnnotations(DenseAnnotation):
    def __init__(self, video, manual_annotations):
        super().__init__()
        self.video = video
        self.annotations = manual_annotations
        self.templates = Templates(video,manual_annotations,size=(64,64))
    def generate(self,starting_index):
        # Validate data
        if self.video is None:
            raise Exception('Video must be provided to generate annotations.')
        # Check if there's enough data
        if len(self.annotations) == 0:
            return
        # Generate annotation for each frame
        num_frames = self.video.frame_count
        try:
            for frame_index in tqdm(range(starting_index,num_frames),desc='Generating annotations'):
                ann = self.search_frame(
                        frame_index,
                        window_size=(128,128))
                self[frame_index] = ann
        except KeyboardInterrupt:
            pass
    def search_frame(self,frame_index,window_size,
            method=cv2.TM_SQDIFF_NORMED):
        width = self.video.width
        height = self.video.height

        # Check if there's a manual annotation for this frame
        if self.annotations[frame_index] is not None:
            return self.annotations[frame_index]

        # Get nearest template
        template = self.templates[frame_index]
        if template is None:
            return None

        # Get a small window to search through
        nearest_coord = self.annotations[frame_index-1]
        if nearest_coord is None:
            nearest_coord = self[frame_index-1]

        frame = self.video.get_frame(frame_index)
        if nearest_coord is not None:
            offset_x = int(nearest_coord[0]*width-window_size[0]/2)
            offset_y = int(nearest_coord[1]*height-window_size[1]/2)
            offset_x = max(offset_x,0)
            offset_x = min(offset_x,width-window_size[0])
            offset_y = max(offset_y,0)
            offset_y = min(offset_y,height-window_size[1])
            window = frame[offset_y:offset_y+window_size[1],offset_x:offset_x+window_size[0],:]
        else:
            offset_x = 0
            offset_y = 0
            window = frame

        # Search for template in frame
        log.info('Searching frame for match')
        res = cv2.matchTemplate(window,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        template_size = template.shape[:2]
        return (
            (top_left[0]+template_size[0]/2+offset_x)/width,
            (top_left[1]+template_size[1]/2+offset_y)/height
        )
