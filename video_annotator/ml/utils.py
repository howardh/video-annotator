import itertools
import torch

def parse_anchor_boxes(coord, vis, normalized_vis=False):
    """ Parse a tensor representing anchor box outputs, and
    return a list of coordinates with visibility prediction for that point
    """
    if len(coord.shape) > 3:
        raise ValueError('Received coord of shape %s. Expected a single coord of shape [2,*,*].' % str(coord.shape))
    if len(vis.shape) > 3:
        raise ValueError('Received vis of shape %s. Expected a single vis of shape [1,*,*].' % str(coord.shape))

    output = []
    boxes = torch.tensor(coord.shape[-2:]) # Number of anchor boxes in each dimension
    box_dims = 1/boxes.float()
    for box in itertools.product(range(boxes[0]),range(boxes[1])):
        # Anchor box indices
        box = torch.tensor(box)
        # Visibility
        if normalized_vis:
            v = vis[0,box[0],box[1]]
        else:
            v = torch.sigmoid(vis[0,box[0],box[1]])
        # Coordinate relative to the upper-left corner of anchor box
        rel_coord = coord[:,box[0],box[1]]
        # Absolute coordinate in [0,1]^2
        abs_coord = (box+rel_coord)*box_dims
        # Save to output
        output.append({
            'coord': abs_coord,
            'vis': v
        })
    return output
