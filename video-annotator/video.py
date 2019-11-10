import cv2

class Video(object):
    def __init__(self,video_file_path):
        self.cap = cv2.VideoCapture(video_file_path)
        if not self.cap.isOpened():
            raise Exception("Error opening video stream or file")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('Number of frames:', self.frame_count)
        print('Frames per second:', self.fps)
        print('Frame width:', self.width)
        print('Frame height:', self.height)

    def get_frame(self, frame_index=None):
        current_frame_index = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_index is not None and frame_index != current_frame_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        return frame

    def close(self):
        self.cap.release()
