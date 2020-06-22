import video_annotator
from video_annotator.annotation import SparseAnnotation, map_annotations

def test_create_sparse_annotation():
    a = SparseAnnotation()

def test_map_annotations_0():
    p = [(0,0),(1,1)]
    c = [(0.01,0.01),(0.99,0.99)]
    output = map_annotations(p,c)
    assert output == [0,1]

def test_map_annotations_1():
    p = [(1,1)]
    c = [(0.01,0.01),(0.99,0.99)]
    output = map_annotations(p,c)
    assert output == [None,0]
