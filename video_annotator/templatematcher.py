import cv2
import numpy as np
from tqdm import tqdm

import logging

log = logging.getLogger(__name__)

def clip(val,min_val,max_val):
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val

class Templates:
    def __init__(self,video,manual_annotations,size):
        self.templates = {}
        self.video = video.clone()
        self.annotations = manual_annotations
        self.size = size
    def __getitem__(self,index):
        # Find closest manual annotation to frame_index
        low,high = self.annotations.nearest_indices(index)
        if self.annotations[low] is None or self.annotations[high] is None:
            return None
        # Use the previous annotation, so adding a new manual annotation only affects future labels.
        nearest_index = low
        # Compute template if needed
        if nearest_index not in self.templates:
            width = self.video.width
            height = self.video.height
            size = self.size
            coord = self.annotations[nearest_index]
            frame = self.video.get_frame(nearest_index)
            x = clip(int(width*coord[0]-size[0]/2),0,width-1)
            y = clip(int(height*coord[1]-size[1]/2),0,height-1)
            template = frame[y:(y+size[1]),x:(x+size[0]),:]
            self.templates[nearest_index] = template
        return self.templates[nearest_index]
    def set_size(self,size):
        self.size = size
        self.clear()
    def clear(self):
        self.templates = {}
