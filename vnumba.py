from numba import njit, prange
import numpy as np

@njit
def step(z, c):
    return z * z + c

@njit
def escape(n, z, c):
    i = 0
    while abs(z) < 2:
        i += 1
        if i > n: break
        z = step(z, c)
    return i

@njit
def escapes(n, z, m, ws, cs):
    return [[[escape(n, z, complex(x, y))
              for x in np.linspace(c.real-w, c.real+w, m)]
              for y in np.linspace(c.imag-w, c.imag+w, m)]
              for w, c in zip(ws, cs)]

@njit(parallel=True)
def escapes_(n, z, m, ws, cs):
    arr = np.empty((len(ws), m, m))
    for i in prange(len(ws)):
        w, c = ws[i], cs[i]
        for j, y in enumerate(np.linspace(c.imag-w, c.imag+w, m)):
            for k, x in enumerate(np.linspace(c.real-w, c.real+w, m)):
                arr[i,j,k] = escape(n, z, complex(x, y))
    return arr