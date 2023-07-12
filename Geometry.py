import numpy as np
from collections import namedtuple


Cube = namedtuple(
    'Cube', 
    ['left_top_inner_corner', 'right_top_inner_corner',
    'left_top_outer_corner', 'right_top_outer_corner',

    'left_bottom_inner_corner', 'right_bottom_inner_corner',
    'left_bottom_outer_corner', 'right_bottom_outer_corner'])

def create_cube(length):
    offset = length/2.0

    return Cube(
        left_top_inner_corner=np.array([-offset, offset, offset]),
        right_top_inner_corner=np.array([offset, offset, offset]),
        left_top_outer_corner=np.array([-offset, offset, -offset]),
        right_top_outer_corner=np.array([offset, offset, -offset]),

        left_bottom_inner_corner=np.array([-offset, -offset, offset]),
        right_bottom_inner_corner=np.array([offset, -offset, offset]),
        left_bottom_outer_corner=np.array([-offset, -offset, -offset]),
        right_bottom_outer_corner=np.array([offset, -offset, -offset]))

def length(cube):
    return np.linalg.norm(cube.right_top_inner_corner - cube.left_top_inner_corner)

def area(cube):
    cube_length = length(cube)
    return cube_length ** 2
