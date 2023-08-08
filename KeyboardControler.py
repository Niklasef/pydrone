import keyboard
from collections import namedtuple


Mapping = namedtuple('Mapping', ['axis', 'dir'])

INPUT_MAGNITUDE = 1.0

def poll_keyboard():
    key_mappings = {
        'a': Mapping('z_rot', -INPUT_MAGNITUDE),
        'd': Mapping('z_rot', INPUT_MAGNITUDE),
        's': Mapping('x_rot', -INPUT_MAGNITUDE),
        'w': Mapping('x_rot', INPUT_MAGNITUDE),
        'left': Mapping('y_rot', -INPUT_MAGNITUDE),
        'right': Mapping('y_rot', INPUT_MAGNITUDE),
        'up': Mapping('y_trans', INPUT_MAGNITUDE),
        'down': Mapping('y_trans', -INPUT_MAGNITUDE)
    }

    pressed_keys = {}

    for key in key_mappings:
        mapping = key_mappings[key]
        pressed_keys[mapping.axis] = 0

    for key in key_mappings:
        if keyboard.is_pressed(key):
            mapping = key_mappings[key]
            pressed_keys[mapping.axis] = mapping.dir
    
    return pressed_keys
