def main(args):
    # Imports are here so we don't need to wait for them to load unecessarily.
    import os
    import tkinter

    from video import Video
    from annotation import Annotations
    from state import State
    import gui

    # Parameters
    video_file_path = args.video_file_path
    annotation_file_path = args.annotation_file_path

    if annotation_file_path is None:
        # Expect the following dir structure:
        # dataset/
        # - videos/
        # - annotations/
        split_path = os.path.split(video_file_path)
        annotation_file_name = split_path[-1].split('.')[0]+'.pkl'
        annotation_file_dir = list(split_path[:-1])+['..','annotations']
        annotation_file_dir = os.path.join(*annotation_file_dir)
        if not os.path.isdir(annotation_file_dir):
            print('Invalid directory structure.')
            return
        annotation_file_path = os.path.join(
                annotation_file_dir,annotation_file_name)

    # Load Video
    video = Video(video_file_path)
    annotations = Annotations(annotation_file_path, video)
    state = State(video,annotations)

    # Create GUI
    gui.App(tkinter.Tk(), state)

    # When everything done, release the video capture object
    video.close()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Video Annotation')
    parser.add_argument('--video_file_path', type=str, required=True,
                        help='Path to the video file to be annotated.')
    parser.add_argument('--annotation_file_path', type=str, required=False,
                        default=None,
                        help='Path to the file where annotations are saved.')
    args = parser.parse_args()
    print(args)

    main(args)
