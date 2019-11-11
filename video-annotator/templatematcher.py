import cv2
import numpy as np
from tqdm import tqdm

import logging

import annotation

log = logging.getLogger(__name__)

class Templates:
    def __init__(self,video,manual_annotations,size):
        self.templates = {}
        self.video = video
        self.annotations = manual_annotations
        self.size = size
    def __getitem__(self,index):
        # Find closest manual annotation to frame_index
        low,high = self.annotations.nearest_indices(index)
        if self.annotations[low] is None or self.annotations[high] is None:
            return None
        if index-low < high-index:
            nearest_index = low
        else:
            nearest_index = high
        # Compute template if needed
        if nearest_index not in self.templates:
            width = self.video.width
            height = self.video.height
            size = self.size
            coord = self.annotations[nearest_index]
            frame = self.video.get_frame(nearest_index)
            x = int(width*coord[0]-size[0]/2)
            y = int(height*coord[1]-size[1]/2)
            template = frame[y:(y+size[1]),x:(x+size[0]),:]
            self.templates[nearest_index] = template
        return self.templates[nearest_index]
    def clear(self):
        self.templates = {}
