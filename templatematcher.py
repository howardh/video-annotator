import cv2
import numpy as np

def generate_annotation(video,frame,template_size=(64,64),annotation_id=None,method=cv2.TM_CCOEFF):
    """ Given a video with annotations, annotate a new frame by
    template matching
    """

    if annotation_id is None:
        annotation_id = list(video.annotations.keys())[0]

    # Get template
    templates = []
    for frame_index,coord in video.annotations[annotation_id].items():
        frame = video.get_frame(frame_index,show_annotations=False)
        width = video.width
        height = video.height
        x = int(width*coord[0]-template_size[0]/2)
        y = int(height*coord[1]-template_size[1]/2)
        template = frame[y:(y+template_size[1]),x:(x+template_size[0]),:]
        templates.append(template)
    mean_template = np.mean(templates,0).astype(np.uint8)

    # Search for template in frame
    res = cv2.matchTemplate(frame,mean_template,cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    return {annotation_id: (top_left[0]/width,top_left[1]/height)}
