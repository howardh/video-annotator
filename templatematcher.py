import cv2
import numpy as np
from tqdm import tqdm

import logging

log = logging.getLogger(__name__)

templates = None
mean_template = None

def compute_templates(video,template_size,annotation_id):
    width = video.width
    height = video.height
    templates = []
    for index,coord in tqdm(video.annotations[annotation_id].items(),desc='Creating template'):
        if coord is None:
            continue
        frame = video.get_frame(index,show_annotations=False)
        x = int(width*coord[0]-template_size[0]/2)
        y = int(height*coord[1]-template_size[1]/2)
        template = frame[y:(y+template_size[1]),x:(x+template_size[0]),:]
        templates.append(template)
        cv2.imwrite('temp/template-%d.png'%index,template)
        break
    mean_template = np.mean(templates,0).astype(np.uint8)
    return templates, mean_template

def generate_annotation(video,frame_index,template_size=(64,64),annotation_id=None,method=cv2.TM_SQDIFF_NORMED,window_size=(256,256)):
    """ Given a video with annotations, annotate a new frame by
    template matching
    """

    global templates, mean_template

    if annotation_id is None:
        annotation_id = list(video.annotations.keys())[0]

    width = video.width
    height = video.height

    # Get template
    log.info('Computing mean template')
    if templates is None or mean_template is None:
        templates, mean_template = compute_templates(
                video,template_size,annotation_id)

    # Get a small window to search through
    nearest_coord = video.generated_annotations[annotation_id][frame_index-1]
    frame = video.get_frame(frame_index,show_annotations=False)
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
    #frame = video.get_frame(frame_index,show_annotations=False)
    res = cv2.matchTemplate(window,mean_template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    return {
        annotation_id: (
            (top_left[0]+template_size[0]/2+offset_x)/width,
            (top_left[1]+template_size[1]/2+offset_y)/height
        )
    }

if __name__=='__main__':
    import script
    video = script.Video('./2019-11-01.webm')
    video.load_annotations('./annotations.pkl')
    frame_index = (3*60+22)*video.fps+25
    print(frame_index)

    # Save Frame
    frame = video.get_frame(frame_index,show_annotations=False)
    cv2.imwrite('frame.png',frame)

    methods = [
            cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED,
            cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
            cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED
    ]
    method_names = [
            'TM_CCOEFF', 'TM_CCOEFF_NORMED',
            'TM_CCORR', 'TM_CCORR_NORMED',
            'TM_SQDIFF', 'TM_SQDIFF_NORMED'
    ]
    method_colours = [
            (255,0,0),(255,0,0),
            (0,255,0),(0,255,0),
            (0,0,255),(0,0,255)
    ]
    for method,method_name,method_colour in zip(methods,method_names,method_colours):
        print(method_name)
        annotation = generate_annotation(video,frame_index,method=method)
        centre = (
                int(annotation[0][0]*video.width),
                int(annotation[0][1]*video.height)
        )
        shifted_centre = (
                int(annotation[0][0]*video.width)+method,
                int(annotation[0][1]*video.height)+method
        )
        cv2.circle(frame, center=centre,
                radius=10, color=method_colour, thickness=5, lineType=8, shift=0)
        cv2.putText(frame, method_name, shifted_centre, cv2.FONT_HERSHEY_SIMPLEX, 1, method_colour, 3)

    cv2.imwrite('frame2.png',frame)
