def forward_derivative(function, x):
    """
    :param function: Function for which to calculate the integral.
    :param x: The point at which the derivative of the function needs to be evaluated.
    :return: The estimated derivative of the function at the given point x.
    """
    h = 1e-8
    return (function(x + h) - function(x)) / h


def backward_derivative(function, x):
    """
    :param function: Function for which to calculate the integral.
    :param x: The point at which the derivative of the function needs to be evaluated.
    :return: The estimated derivative of the function at the given point x.
    """
    h = 1e-8
    return (function(x) - function(x - h)) / h


def centered_derivative(function, x) -> 'Values of the centered derivative':
    """
    :param function: Function for which to calculate the integral.
    :param x: The point at which the derivative of the function needs to be evaluated.
    :return: The estimated derivative of the function at the given point x.
    """
    h = 1e-5
    return (function(x + h / 2) - function(x - h / 2)) / h
