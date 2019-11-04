import cv2
import numpy as np

import logging

log = logging.getLogger(__name__)

templates = None
mean_template = None

def compute_templates(video,template_size,annotation_id):
    width = video.width
    height = video.height
    templates = []
    for index,coord in video.annotations[annotation_id].items():
        frame = video.get_frame(index,show_annotations=False)
        x = int(width*coord[0]-template_size[0]/2)
        y = int(height*coord[1]-template_size[1]/2)
        template = frame[y:(y+template_size[1]),x:(x+template_size[0]),:]
        templates.append(template)
        cv2.imwrite('temp/template-%d.png'%index,template)
    mean_template = np.mean(templates,0).astype(np.uint8)
    return templates, mean_template

def generate_annotation(video,frame_index,template_size=(64,64),annotation_id=None,method=cv2.TM_SQDIFF_NORMED):
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

    ## Compute Prior (New label should be near interpolated point)
    #log.info('Computing prior')
    #nearest_coord = video.get_annotation(frame_index,annotation_id)
    #mu_x = int(width*nearest_coord[0]-template_size[0]/2)
    #mu_y = int(height*nearest_coord[1]-template_size[1]/2)
    #print(mu_x,mu_y)
    #std = 100
    #coef = 1/(np.sqrt(2*np.pi)*std)
    #gauss = lambda y,x: coef*np.exp(-((x-mu_x)**2+(y-mu_y)**2)/(2*std**2))
    #prior = np.fromfunction(
    #        gauss,
    #        shape=(
    #            int(height-template_size[1]+1),
    #            int(width-template_size[0]+1)),
    #        dtype=np.float)
    #prior = prior/prior.max()*0.9
    #print(prior)
    #print(prior.min(), prior.max())
    #cv2.imwrite('prior.png',prior*255)
    #print(prior*255)

    # Search for template in frame
    log.info('Searching frame for match')
    frame = video.get_frame(frame_index,show_annotations=False)
    res = cv2.matchTemplate(frame,mean_template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res*(1-prior))
        top_left = min_loc
    else:
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res*prior)
        top_left = max_loc

    return {
        annotation_id: (
            (top_left[0]+template_size[0]/2)/width,
            (top_left[1]+template_size[1]/2)/height
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
