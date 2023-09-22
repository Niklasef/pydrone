import numpy as np
from collections import namedtuple
from CoordinateSystem import transform_to_global

NavPoint = namedtuple('NavPoint', ['coordinate_system', 'position'])

def calculate_distance_to_optimal_path(drone_position, path_point1, path_point2):
    v = path_point2 - path_point1  # vector along the line segment
    w = drone_position - path_point1  # vector from path_point1 to drone_position
    
    c1 = np.dot(w, v)
    if c1 <= 0:  # point is closer to path_point1
        return np.linalg.norm(drone_position - path_point1)
    
    c2 = np.dot(v, v)
    if c2 <= c1:  # point is closer to path_point2
        return np.linalg.norm(drone_position - path_point2)
    
    b = c1 / c2
    point_on_segment = path_point1 + b * v  # projection of the drone_position onto the segment
    return np.linalg.norm(drone_position - point_on_segment)

def nav_error(
    ack_error,
    start_nav_point,
    end_nav_point,
    drone,
    delta_time):
    start_nav_point_global = transform_to_global(
        start_nav_point.coordinate_system,
        start_nav_point.position)
    end_nav_point_global = transform_to_global(
        end_nav_point.coordinate_system,
        end_nav_point.position)
    drone.coordinate_system.origin

    distance_to_nav_point = np.linalg.norm(
        drone.coordinate_system.origin - end_nav_point_global)

    distance_to_optimal_path = calculate_distance_to_optimal_path(
        drone.coordinate_system.origin,
        start_nav_point_global,
        end_nav_point_global)

    return ack_error + ((distance_to_optimal_path * delta_time) / distance_to_nav_point) if distance_to_nav_point > 0.2 else ack_error
