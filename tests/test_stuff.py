import script
from script import interpolate_annotations

def test_interpolate_annotations_empty():
    points = {}
    output = interpolate_annotations(points)
    assert output == []

def test_interpolate_annotations_1_point():
    points = {1: (0,0)}
    output = interpolate_annotations(points)
    assert output == [None, None]

def test_interpolate_annotations_2_points():
    points = {1: (0,0), 2: (1,1)}
    output = interpolate_annotations(points)
    assert output == [None, (0,0), (1,1)]
