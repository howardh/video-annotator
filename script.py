import cv2
import time
import pickle
import os
import argparse
import tkinter

import gui

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
            output[start+i] = (int(points[start][0]+diff_per_frame[0]*i),int(points[start][1]+diff_per_frame[1]*i))
    return output

def play_video(cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

class Video(object):
    def __init__(self,video_file_path):
        self.cap = cv2.VideoCapture(video_file_path)
        if not self.cap.isOpened():
            raise Exception("Error opening video stream or file")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('Number of frames:', self.frame_count)
        print('Frames per second:', self.fps)
        print('Frame width:', self.width)
        print('Frame height:', self.height)

    def get_frame(self, frame_index=None, show_annotations=False):
        if frame_index is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if show_annotations:
            for ann_id,interp_ann in self.interpolated_annotations.items():
                if frame_index < len(interp_ann) and interp_ann[frame_index] is not None:
                    cv2.circle(frame, center=interp_ann[frame_index],
                            radius=10, color=(0,255,0), thickness=5, lineType=8, shift=0)
        return frame

    def add_annotation(self, frame_index, annotation_id, annotation):
        self.annotations[annotation_id][frame_index] = annotation
        self.interpolated_annotations[annotation_id] = interpolate_annotations(self.annotations[annotation_id])

    def remove_annotation(self, frame_index, annotation_id):
        if frame_index not in self.annotations[annotation_id]:
            print('No keyframe selected')
        del self.annotations[annotation_id][frame_index]
        self.interpolated_annotations[annotation_id] = interpolate_annotations(self.annotations[annotation_id])

    def load_annotations(self, annotation_file_path):
        # Load data
        if os.path.isfile(annotation_file_path):
            with open(annotation_file_path, 'rb') as f:
                self.annotations = pickle.load(f)
        else:
            self.annotations = {}
        # Process data
        self.interpolated_annotations = {}
        for k,v in self.annotations.items():
            self.interpolated_annotations[k] = interpolate_annotations(v)
    
    def save_annotations(self, annotation_file_path):
        with open(annotation_file_path, 'wb') as f:
            pickle.dump(video.annotations, f)

    def close(self):
        self.cap.release()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Video Annotation')
    parser.add_argument('--video_file_path', type=str, required=False,
                        default='/home/howard/Videos/Gym/2019-08-02/2019-08-02.webm',
                        help='Path to the video file to be annotated.')
    parser.add_argument('--annotation_file_path', type=str, required=False,
                        default='./annotations.pkl',
                        help='Path to the file where annotations are saved.')
    parser.add_argument('--annotation_id', type=int, required=False,
                        default=None,
                        help='ID of annotation to work on.')
    args = parser.parse_args()
    print(args)

    # Parameters
    video_file_path = args.video_file_path
    annotation_file_path = args.annotation_file_path
    annotation_id = args.annotation_id

    # Load Video
    video = Video(video_file_path)
    video.load_annotations(annotation_file_path)

    # Create GUI
    gui.App(tkinter.Tk(), video)

    # Save annotations
    video.save_annotations(annotation_file_path)

    # When everything done, release the video capture object
    video.close()
    # Closes all the frames
    cv2.destroyAllWindows()
