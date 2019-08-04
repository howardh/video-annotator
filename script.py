import cv2
import time
import pickle
import os

# Parameters
video_file_path = '/home/howard/Videos/Gym/2019-08-02/2019-08-02.webm'
annotation_file_path = './annotations.pkl'
annotation_id = 0
all_annotations = {}

cap = cv2.VideoCapture(video_file_path)

if (cap.isOpened()== False):
    print("Error opening video stream or file")

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames_per_second = cap.get(cv2.CAP_PROP_FPS)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('Number of frames:', num_frames)
print('Frames per second:', frames_per_second)
print('Frame width:', frame_width)
print('Frame height:', frame_height)

# Load annotations if they exist
if os.path.isfile(annotation_file_path):
    with open(annotation_file_path, 'rb') as f:
        all_annotations = pickle.load(f)
else:
    all_annotations = {}
# Load annotation with given ID
if annotation_id in all_annotations:
    points = all_annotations[annotation_id]
    print('Loaded points',points)
else:
    points = {}
    print('No points with provided annotation ID',annotation_id)
# Interpolated points
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
        for i in range(end-start):
            output[start+i] = (int(points[start][0]+diff_per_frame[0]*i),int(points[start][1]+diff_per_frame[1]*i))
    return output
interpolated_points = interpolate_annotations(points)
current_frame = 0
cv2.namedWindow('Frame')
def set_annotation_coordinate(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points[current_frame] = (x,y)
        print('Saved point', current_frame, x, y)
cv2.setMouseCallback('Frame', set_annotation_coordinate)
# Loop through frames
for current_frame in range(num_frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if current_frame in points:
        cv2.circle(frame, center=points[current_frame], radius=10, color=(0,255,0), thickness=5, lineType=8, shift=0)
    if current_frame < len(interpolated_points) and interpolated_points[current_frame] is not None:
        cv2.circle(frame, center=interpolated_points[current_frame], radius=10, color=(0,255,0), thickness=5, lineType=8, shift=0)
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        while cv2.waitKey(0) & 0xFF != ord(' '):
            pass
all_annotations[annotation_id] = points
with open(annotation_file_path, 'wb') as f:
    pickle.dump(all_annotations, f)

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
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
