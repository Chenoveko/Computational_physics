def forward_derivative(f: 'function', x: 'points') -> 'Values of the forward derivative':
    h = 1e-8
    return (f(x+h) - f(x))/h

def backward_derivative(f: 'function', x: 'points') -> 'Values of the backward derivative':
    h = 1e-8
    return (f(x) - f(x-h))/h

def centered_derivative(f: 'function', x: 'points') -> 'Values of the centered derivative':
    h = 1e-5
    return (f(x+h/2) - f(x-h/2))/h