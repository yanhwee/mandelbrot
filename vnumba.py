import numba
import numpy as np

@numba.njit
def step(z, c):
    return z * z + c

@numba.njit
def escape(n, z, c):
    i = 0
    while abs(z) < 2:
        i += 1
        if i > n: break
        z = step(z, c)
    return i

@numba.njit
def escapes(n, z, m, ws, cs):
    return [[[escape(n, z, complex(x, y))
              for x in np.linspace(c.real-w, c.real+w, m)]
              for y in np.linspace(c.imag-w, c.imag+w, m)]
              for w, c in zip(ws, cs)]