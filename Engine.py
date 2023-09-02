from collections import namedtuple
import numpy as np


EngineSpec = namedtuple('EngineSpec', [
    'force_curve',
    'torque_curve',
    'response_time'])
EngineOutput = namedtuple('EngineOutput', ['force', 'torque'])

def init_engine_spec():
    force_curve = np.array([
        (0.0, 0.0),
        (0.2, 1.0),
        (0.4, 2.0),
        (0.6, 3.0),
        (0.8, 4.0),
        (1.0, 5.0)])
    torque_curve = np.array([
        (0.0, 0.0),
        (0.2, 0.1),
        (0.4, 0.2),
        (0.6, 0.3),
        (0.8, 0.4),
        (1.0, 0.5)])
    return EngineSpec(
        force_curve,
        torque_curve,
        1)

def select_point(curve, level):
    lower_points = list(filter(lambda p: p[0] <= level, curve))
    lower_bound = max(lower_points, key=lambda p: p[0])

    upper_points = list(filter(lambda p: p[0] > level, curve))
    upper_bound = max(upper_points, key=lambda p: p[0]) if upper_points else None

    if upper_bound is None:
        return lower_bound[1]

    # Linear interpolation
    slope = (upper_bound[1] - lower_bound[1]) / (upper_bound[0] - lower_bound[0])
    interpolated_value = slope * (level - lower_bound[0]) + lower_bound[1]

    return interpolated_value

def interpolate_response_time(start_level, end_level, delta_time, engine_spec):
    fraction_of_response = delta_time / engine_spec.response_time
    return start_level + (end_level - start_level) * fraction_of_response


def engine_output(engine_spec, input_levels):
    assert input_levels.shape == (4,), "input_levels must be a 1-dimensional array of length 4"
    assert 0 <= input_levels[0] <= 1, "input_level must be between 0 and 1"
    assert 0 <= input_levels[1] <= 1, "input_level must be between 0 and 1"
    assert 0 <= input_levels[2] <= 1, "input_level must be between 0 and 1"
    assert 0 <= input_levels[3] <= 1, "input_level must be between 0 and 1"

    forces = [select_point(engine_spec.force_curve, level) for level in input_levels]
    torques = [select_point(engine_spec.torque_curve, level) for level in input_levels]

    return [
        EngineOutput(forces[0], -torques[0]),
        EngineOutput(forces[1], torques[1]),
        EngineOutput(forces[2], -torques[2]),
        EngineOutput(forces[3], torques[3])
    ]

def engine_output_with_response(
    engine_spec,
    prev_input_levels,
    input_levels,
    delta_time):
    interpolated_levels = np.array([
        interpolate_response_time(
            prev_input_levels[i],
            input_levels[i],
            delta_time,
            engine_spec) 
        for i in range(4)
    ])
    return engine_output(engine_spec, interpolated_levels)


def engine_max_force(engine_spec):
    return max(engine_spec.force_curve, key=lambda p: p[0])[1]
