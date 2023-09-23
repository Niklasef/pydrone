import numpy as np
from collections import namedtuple
from CoordinateSystem import transform_to_global, euler_angles

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
    acc_error,
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


    if distance_to_nav_point < 0.5:  # Return early if too close to the nav point to avoid large normalized errors.
        return (acc_error, True)

    # print(distance_to_nav_point)
    nav_distance = np.linalg.norm(
        start_nav_point_global - end_nav_point_global)

    distance_to_optimal_path = calculate_distance_to_optimal_path(
        drone.coordinate_system.origin,
        start_nav_point_global,
        end_nav_point_global)

    drone_heading = euler_angles(drone.coordinate_system)[2]
    vector = end_nav_point_global - start_nav_point_global  # vector pointing from start to end
    desired_heading = np.arctan2(vector[0], vector[2])  # atan2(x, z) gives the angle of a vector in the xz-plane from the positive z-axis

    heading_error = drone_heading - desired_heading
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
    heading_error = np.abs(heading_error)
    # Weight the heading_error, calculate the combined error and normalize by distance_to_nav_point
    heading_weight = 0.1
    combined_error = (distance_to_optimal_path + (heading_weight * heading_error)) * delta_time
    normalized_error = combined_error / nav_distance
    
    return (acc_error + normalized_error, False)
