import numpy as np
from collections import namedtuple


Block = namedtuple(
    'Block', 
    ['left_top_outer_corner', 'right_top_outer_corner',
    'left_top_inner_corner', 'right_top_inner_corner',

    'left_bottom_outer_corner', 'right_bottom_outer_corner',
    'left_bottom_inner_corner', 'right_bottom_inner_corner'])

def create_block(length, width, height):
    l_offset = length/2.0
    w_offset = width/2.0
    h_offset = height/2.0

    return Block(
        left_top_outer_corner=np.array([-w_offset, h_offset, l_offset]),
        right_top_outer_corner=np.array([w_offset, h_offset, l_offset]),
        left_top_inner_corner=np.array([-w_offset, h_offset, -l_offset]),
        right_top_inner_corner=np.array([w_offset, h_offset, -l_offset]),

        left_bottom_outer_corner=np.array([-w_offset, -h_offset, l_offset]),
        right_bottom_outer_corner=np.array([w_offset, -h_offset, l_offset]),
        left_bottom_inner_corner=np.array([-w_offset, -h_offset, -l_offset]),
        right_bottom_inner_corner=np.array([w_offset, -h_offset, -l_offset]))

def length(block):
    return np.linalg.norm(block.right_top_outer_corner - block.right_top_inner_corner)

def width(block):
    return np.linalg.norm(block.right_top_outer_corner - block.left_top_outer_corner)

def height(block):
    return np.linalg.norm(block.right_top_outer_corner - block.right_bottom_outer_corner)

def inertia(block, mass):
    L = length(block)
    W = width(block)
    H = height(block)

    I_x = (1/12) * mass * (H**2 + L**2)
    I_y = (1/12) * mass * (W**2 + L**2)
    I_z = (1/12) * mass * (W**2 + H**2)

    return [I_x, I_y, I_z]

# TODO: refactor to one area function
def area_x(block):
    return length(block) * height(block)

def area_y(block):
    return length(block) * width(block)

def area_z(block):
    return width(block) * height(block)
