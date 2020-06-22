import cv2
import time
import pickle
import os
from tqdm import tqdm
from collections import defaultdict
import logging

import numpy as np

import torch
import torchvision

import video_annotator
from video_annotator.templatematcher import Templates
from video_annotator.ml.model import Net, Net2
from video_annotator.ml.transforms import Scale, RandomCrop, CentreCrop, Normalize
from video_annotator.ml.utils import parse_anchor_boxes

log = logging.getLogger(__name__)

class Annotations():
    def __init__(self, file_path, video=None):
        self.file_path = file_path
        self.video = video
        self.annotations = {}
        self.predicted = PredictedAnnotations(video)
        self.predicted2 = PredictedAnnotations2(video)

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
                    'template_matched': ann.template_matched.data,
                    'optical_flow': ann.optical_flow.data,
                    'template_size': ann.get_template_size(),
                    'window_size': ann.get_window_size()
            }
        with open(annotation_file_path, 'wb') as f:
            pickle.dump(output, f)
        print('saving to', annotation_file_path)
    def render(self, frame, frame_index, num_frames=100):
        height,width,_ = frame.shape
        for ann_id in self.annotations.keys():
            ann = self[ann_id][frame_index]
            manu_ann = ann['manual']
            temp_size = self[ann_id].get_template_size()[0]
            win_size = self[ann_id].get_window_size()[0]
            # Manual annotations
            if manu_ann is not None:
                centre = manu_ann
                cx,cy = (int(centre[0]*width),
                        int(centre[1]*height))
                cs = int(temp_size//2) # Cross size
                ws = int(win_size//2) # Window size
                cv2.line(frame,(cx-cs,cy),(cx+cs,cy),
                        color=(0,255,0),thickness=1)
                cv2.line(frame,(cx,cy-cs),(cx,cy+cs),
                        color=(0,255,0),thickness=1)
                cv2.rectangle(frame,rec=(cx-ws,cy-ws,ws*2,ws*2),
                        color=(0,255,0),thickness=1)
            # Generated annotations
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
            # Path from generated annotatoins
            gen_path = self[ann_id].template_matched[frame_index-num_frames:frame_index]
            for c0,c1 in zip(gen_path,gen_path[1:]):
                if c0 is None or c1 is None:
                    continue
                c0 = (int(c0[0]*width),int(c0[1]*height))
                c1 = (int(c1[0]*width),int(c1[1]*height))
                cv2.line(frame,c0,c1,color=(255,0,0),thickness=2)
            # Optical flow annotations
            of_ann = ann['optical_flow']
            if of_ann is not None:
                centre = of_ann
                cx,cy = (int(centre[0]*width),
                        int(centre[1]*height))
                cs = 10 # Cross size
                cv2.line(frame,(cx-cs,cy),(cx+cs,cy),
                        color=(0,255,0),thickness=10)
                cv2.line(frame,(cx,cy-cs),(cx,cy+cs),
                        color=(0,255,0),thickness=10)
        # CNN Prediction
        self.predicted.render(frame, frame_index, (255,0,255))
        # CNN Prediction with anchor boxes
        self.predicted2.render(frame, frame_index, (255,255,255))
        return frame

class Annotation():
    def __init__(self, video):
        self.manual = SparseAnnotation()
        self.template_matched = TemplateMatchedAnnotations(video,self.manual)
        self.optical_flow = OpticalFlowAnnotations(video,self.manual)
    def __getitem__(self, frame_index):
        return {
            'manual': self.manual[frame_index],
            'template_matched': self.template_matched[frame_index],
            'optical_flow': self.optical_flow[frame_index]
        }
    def __setitem__(self, frame_index, annotation):
        self.manual[frame_index] = annotation
    def __delitem__(self, frame_index):
        del self.manual[frame_index]
    def load(self,data):
        self.manual.data = data['manual']
        self.template_matched.data = data['template_matched']
        if 'optical_flow' in data:
            self.optical_flow.data = data['optical_flow']
        self.set_window_size(data.pop('window_size',(128,128)))
        self.set_template_size(data.pop('template_size',(64,64)))
    def set_window_size(self,size):
        if type(size) is int:
            self.template_matched.window_size = (size,size)
        elif type(size) is tuple:
            self.template_matched.window_size = size
    def set_template_size(self,size):
        if type(size) is int:
            self.template_matched.templates.set_size((size,size))
        elif type(size) is tuple:
            self.template_matched.templates.set_size(size)
    def get_window_size(self):
        return self.template_matched.window_size
    def get_template_size(self):
        return self.template_matched.templates.size

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
        self.video = video.clone()
        self.annotations = manual_annotations
        self.templates = Templates(video,manual_annotations,size=(64,64))
        self.window_size = (128,128)
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
                        window_size=self.window_size)
                self[frame_index] = ann
        except KeyboardInterrupt:
            pass
    def generate2(self,starting_index):
        # Validate data
        if self.video is None:
            raise Exception('Video must be provided to generate annotations.')
        # Check if there's enough data
        if len(self.annotations) == 0:
            return
        # Generate annotation for each frame
        num_frames = self.video.frame_count
        funcs = []
        def foo(i):
            frame_index = i+starting_index
            ann = self.search_frame(
                    frame_index,
                    window_size=self.window_size)
            self[frame_index] = ann
        for frame_index in tqdm(range(starting_index,num_frames),desc='Generating annotations'):
            funcs.append(foo)
        return funcs
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

class OpticalFlowAnnotations(DenseAnnotation):
    def __init__(self, video, manual_annotations):
        super().__init__()
        self.video = video.clone()
        self.annotations = manual_annotations
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
                ann = self.search_frame(frame_index)
                self[frame_index] = ann
        except KeyboardInterrupt:
            pass
    def generate2(self,starting_index):
        # Validate data
        if self.video is None:
            raise Exception('Video must be provided to generate annotations.')
        # Check if there's enough data
        if len(self.annotations) == 0:
            return
        # Generate annotation for each frame
        num_frames = self.video.frame_count
        funcs = []
        def foo(i):
            frame_index = i+starting_index
            ann = self.search_frame(frame_index)
            self[frame_index] = ann
        for frame_index in tqdm(range(starting_index,num_frames),desc='Generating annotations'):
            funcs.append(foo)
        return funcs
    def search_frame(self,frame_index):
        # See https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
        if self.annotations[frame_index] is not None:
            return self.annotations[frame_index]

        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        size = np.array([self.video.width, self.video.height],dtype=np.float32)
        prev_img = self.video.get_frame(frame_index-1)
        prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        next_img = self.video.get_frame(frame_index)
        next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
        prev_pts = np.array(self[frame_index-1],dtype=np.float32).reshape(-1,1,2)*size
        p, st, err = cv2.calcOpticalFlowPyrLK(prev_img, next_img, prev_pts, None)
        p = p/size
        return tuple(p.squeeze().tolist())

class PredictedAnnotations(DenseAnnotation):
    def __init__(self, video, model=Net, checkpoint='checkpoints/checkpoint-2000.pt'):
        super().__init__()
        self.video = video.clone()
        with open(checkpoint,'rb') as f:
            checkpoint = torch.load(f,map_location=torch.device('cpu'))
        self.net = model()
        self.net.load_state_dict(checkpoint['model'])
        self.net.eval()
        print('Model loaded')
    def generate(self,starting_index):
        # Validate data
        if self.video is None:
            raise Exception('Video must be provided to generate annotations.')
        # Generate annotation for each frame
        num_frames = self.video.frame_count
        try:
            for frame_index in tqdm(range(starting_index,num_frames),desc='Generating annotations'):
                ann = self.search_frame(frame_index)
                self[frame_index] = ann
        except KeyboardInterrupt:
            pass
    def generate2(self,starting_index):
        # Validate data
        if self.video is None:
            raise Exception('Video must be provided to generate annotations.')
        # Generate annotation for each frame
        num_frames = self.video.frame_count
        funcs = []
        def foo(i):
            frame_index = i+starting_index
            ann = self.search_frame(frame_index)
            self[frame_index] = ann
        for frame_index in tqdm(range(starting_index,num_frames),desc='Generating annotations'):
            funcs.append(foo)
        return funcs
    def search_frame(self,frame_index):
        width = self.video.width
        height = self.video.height

        frame = self.video.get_frame(frame_index)
        # Scale
        size = Scale.get_params(frame,int((300+224)/2))
        frame = Scale.scale_image(size,frame)
        # Crop
        i,j,_,_ = CentreCrop.get_params(frame,(224,224))
        frame = CentreCrop.crop_image((i,j), (224,224), frame)
        # To Tensor
        to_tensor = torchvision.transforms.ToTensor()
        frame = to_tensor(frame)
        # Normalize
        normalize = Normalize()
        frame = normalize.transform(frame)

        frame = frame.view(1,3,224,224)

        # Search for template in frame
        coord,vis = self.net(frame)
        if vis > 0.5:
            # Rescale coordinates back to original frame
            s1 = torch.tensor([size[0],size[1]])
            s2 = torch.tensor([224,224])
            coord = (coord*s2+(s1-s2)/2)/s1
            coord = (coord[0][0].item(), coord[0][1].item())
            return coord
        else:
            return None
    def render(self, frame, frame_index, colour, cross_size=10, path_len=100):
        if frame_index >= len(self):
            return

        height,width,_ = frame.shape

        # Annotation path
        pred = self[frame_index-path_len:frame_index]
        for c0,c1 in zip(pred,pred[1:]):
            if c0 is None or c1 is None:
                continue
            c0 = (int(c0[0]*width),int(c0[1]*height))
            c1 = (int(c1[0]*width),int(c1[1]*height))
            cv2.line(frame,c0,c1,color=colour,thickness=2)
        # Annotation Keypoint
        if self[frame_index] is not None:
            centre = self[frame_index]
            cx,cy = (int(centre[0]*width),
                    int(centre[1]*height))
            cs = cross_size
            cv2.line(frame,(cx-cs,cy),(cx+cs,cy),
                    color=colour,thickness=1)
            cv2.line(frame,(cx,cy-cs),(cx,cy+cs),
                    color=colour,thickness=1)

def map_annotations(prev,curr,threshold=np.exp(-0.01)):
    if len(curr) == 0 or len(prev) == 0:
        return [None]*len(curr)

    score = [[None]*len(curr) for _ in range(len(prev))]
    for pi,p in enumerate(prev):
        p = np.array(p)
        for ci,c in enumerate(curr):
            c = np.array(c)
            score[pi][ci] = np.exp(-((c-p)**2).sum())
    score = np.array(score)
    mapping = [None]*len(curr)
    for _ in range(len(curr)):
        m = score.max()
        if m < threshold:
            break
        pi,ci = np.where(score == m)
        mapping[ci[0]] = pi[0]
        score[:,ci[0]] = 0
        score[pi[0],:] = 0
    return mapping

class PredictedAnnotations2(PredictedAnnotations):
    def __init__(self, video, model=Net2, checkpoint='checkpoints/checkpoint2-5500.pt'):
        super().__init__(video, model, checkpoint)
        self.mapping = [None]*video.frame_count
    def postprocess(self):
        self.map_path()
        self.smooth_path()
    def map_path(self):
        """ Compute the path of a keypoint. """
        for i in tqdm(range(1,len(self.data)), desc='Computing Mapping'):
            prev = self.data[i-1]
            curr = self.data[i]
            self.mapping[i] = map_annotations(prev,curr)
    def smooth_path(self):
        output = [self.data[0]]
        zipped_data = [
            self.data,
            self.data[1:],
            self.data[2:],
            self.mapping,
            self.mapping[1:],
            self.mapping[2:]
        ]
        for c0,c1,c2,m0,m1,m2 in zip(*zipped_data):
            smoothed = []
            for i1,c in enumerate(c1):
                # Prev point
                i0 = m1[i1]
                if i0 is None:
                    smoothed.append(c)
                    continue
                # Next point
                if i1 not in m2:
                    smoothed.append(c)
                    continue
                i2 = m2.index(i1)
                # Coords
                prev_c = np.array(c0[i0])
                next_c = np.array(c2[i2])
                c = np.array(c)
                # Average
                smoothed.append((prev_c+next_c+c)/3)
            output.append(smoothed)
        output.append(self.data[-1])
        self.data = output
    def search_frame_1(self,frame_index):
        width = self.video.width
        height = self.video.height

        frame = self.video.get_frame(frame_index)
        # Params
        scale_size = int((300+224)/2)
        crop_size = (224,224)
        # Scale
        size = Scale.get_params(frame,scale_size)
        frame = Scale.scale_image(size,frame)
        # Crop
        i,j,_,_ = RandomCrop.get_params(frame, crop_size)
        frame = CentreCrop.crop_image((i,j), crop_size, frame)
        crop_top_left = (j,i)
        # To Tensor
        to_tensor = torchvision.transforms.ToTensor()
        frame = to_tensor(frame)
        # Normalize
        normalize = Normalize()
        frame = normalize.transform(frame)

        frame = frame.view(1,3,224,224)

        # Search for template in frame
        output = []
        coord,vis = self.net(frame)
        for p in parse_anchor_boxes(coord[0],vis[0]):
            if p['vis'] < 0.5:
                continue
            # Rescale coordinates back to original frame
            s1 = torch.tensor([size[0],size[1]]) # Frame size after resizing
            s2 = torch.tensor(crop_size) # Frame size after cropping
            tl = torch.tensor(crop_top_left) # Top-left coordinate of cropped window relative to original frame
            coord = (p['coord']*s2+tl)/s1
            coord = (coord[0].item(), coord[1].item())
            output.append(coord)
        return output
    def search_frame(self,frame_index):
        threshold = 0.01
        results = [self.search_frame_1(frame_index) for _ in range(5)]
        groupings = []
        for coords in results:
            if len(groupings) == 0:
                for c in coords:
                    groupings.append([c])
            means = [np.mean(g,axis=0) for g in groupings]
            for c in coords:
                dist_from_means = ((means-np.array(c))**2).sum(axis=1)
                if dist_from_means.min() > threshold:
                    groupings.append([c])
                else:
                    groupings[dist_from_means.argmin()].append(c)
        means = [np.mean(g,axis=0) for g in groupings]
        return means
    def render(self, frame, frame_index, colour, cross_size=10, path_len=100):
        if frame_index >= len(self):
            return

        height,width,_ = frame.shape
        for ci,centre in enumerate(self.data[frame_index]):
            cx,cy = (int(centre[0]*width),
                    int(centre[1]*height))
            cs = int(cross_size//2) # Cross size
            cv2.line(frame,(cx-cs,cy),(cx+cs,cy),
                    color=colour,thickness=5)
            cv2.line(frame,(cx,cy-cs),(cx,cy+cs),
                    color=colour,thickness=5)

            # Draw Path
            i1 = None
            i0 = ci
            for path_index in range(path_len):
                if frame_index-path_index-1 < 0:
                    break
                if frame_index >= len(self.mapping):
                    break
                i1 = i0
                i0 = self.mapping[frame_index-path_index][i1]
                if i0 is None or i1 is None:
                    break
                c1 = self.data[frame_index-path_index][i1]
                c0 = self.data[frame_index-path_index-1][i0]
                if c0 is None or c1 is None:
                    break
                c0 = (int(c0[0]*width),int(c0[1]*height))
                c1 = (int(c1[0]*width),int(c1[1]*height))
                cv2.line(frame,c0,c1,color=colour,thickness=2)
