import argparse
import tkinter

from video import Video
from annotation import Annotations
import gui

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Video Annotation')
    parser.add_argument('--video_file_path', type=str, required=True,
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
    annotations = Annotations(annotation_file_path, video)

    # Create GUI
    gui.App(tkinter.Tk(), video, annotations)

    # When everything done, release the video capture object
    video.close()
