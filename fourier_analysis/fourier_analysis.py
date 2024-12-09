import numpy as np


def discrete_fourier_transform(y):
    N = len(y)

    c = np.zeros(N // 2 + 1, complex)

    for k in range(N // 2 + 1):
        for n in range(N):
            c[k] += y[n] * np.exp(-2j * np.pi * k * n / N)
    return c


def discrete_cosen_transform(y):
    N = len(y)
    y2 = np.empty(2 * N, float)

    for n in range(N):
        y2[n] = y[n]
        y2[2 * N - 1 - n] = y[n]

    c = np.fft.rfft(y2)
    phi = np.exp(-1j * np.pi * np.arange(N) / (2 * N))

    return np.real(phi * c[:N])


def inverse_discrete_cosen_transform(a):
    N = len(a)

    c = np.empty(N, complex)

    phi = np.exp(1j * np.pi * np.arange(N) / (2 * N))
    c[:N] = phi * a
    return np.fft.irfft(c)[:N]
