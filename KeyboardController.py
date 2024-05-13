import keyboard
from collections import namedtuple

Mapping = namedtuple('Mapping', ['axis', 'dir'])

INPUT_MAGNITUDE = 1.0

def poll_keyboard():
    key_mappings = {
        'left': Mapping('z_rot', -INPUT_MAGNITUDE),
        'right': Mapping('z_rot', INPUT_MAGNITUDE),
        'down': Mapping('x_rot', -INPUT_MAGNITUDE),
        'up': Mapping('x_rot', INPUT_MAGNITUDE),
        'a': Mapping('y_rot', -INPUT_MAGNITUDE),
        'd': Mapping('y_rot', INPUT_MAGNITUDE),
        'w': Mapping('y_trans', INPUT_MAGNITUDE),
        's': Mapping('y_trans', -INPUT_MAGNITUDE)
    }

    # Adding number keys for debugging
    number_keys = {str(i): i for i in range(10)}

    pressed_keys = {}

    # Initialize all controlled axes to zero
    for key in key_mappings:
        mapping = key_mappings[key]
        pressed_keys[mapping.axis] = 0
    
    # Check and update values based on key press status
    for key in key_mappings:
        if keyboard.is_pressed(key):
            mapping = key_mappings[key]
            pressed_keys[mapping.axis] += mapping.dir

    # Debugging keys
    debug_keys_pressed = []
    for key in number_keys:
        if keyboard.is_pressed(key):
            debug_keys_pressed.append(number_keys[key])

    # Append debugging information
    pressed_keys['debug'] = debug_keys_pressed

    return pressed_keys
