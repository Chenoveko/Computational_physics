import numpy as np
import inspect


def polares_2_cartesianas(r, theta):
    """
    :param r: The radius, representing the distance from the origin in polar coordinates.
    :param theta: The angle, in radians, representing the orientation from the positive x-axis in polar coordinates.
    :return: A tuple representing the Cartesian coordinates (x, y),
             where x is the horizontal distance and y is the vertical
             distance from the origin.
        - x: The horizontal distance from the origin.
        - y: The vertical distance from the origin.
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def cartesianas_2_polares(x, y):
    """
    :param x: Position on the x-axis of the Cartesian plane.
    :param y: Position on the y-axis of the Cartesian plane.
    :return:
        - r: The radius of the point.
        - theta: The angle of the point in radians.
    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta


def esfericas_2_cartesianas(r, theta, phi):
    """
    :param r: Radial distance from the origin to the point, in meters
    :param theta: Colatitude angle in radians, ranging from 0 to π
    :param phi: Azimuth angle in radians, ranging from 0 to 2π
    :return: Tuple representing the Cartesian coordinates (x, y, z)
        - x: Horizontal distance from the origin
        - y: Vertical distance from the origin
        - z: Distance from the origin to the point along the z-axis
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cilindricas_2_cartesianas(rho, phi, z):
    """
    :param rho: Radial distance from the origin to the point, in meters.
    :param phi: Azimuth angle in radians, ranging from 0 to 2π.
    :param z: Height above the xy plane, in meters.
    :return: Tuple representing the Cartesian coordinates (x, y, z)
        - x: Horizontal distance from the origin
        - y: Vertical distance from the origin
        - z: Distance from the origin to the point along the z-axis
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = z
    return x, y, z


def count_arguments(f):
    """
    :param f: The function whose arguments are to be counted.
    :return num_arguments: The number of arguments of the provided function.
    """
    signature = inspect.signature(f)
    num_arguments = len(signature.parameters)
    return num_arguments
