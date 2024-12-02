from itertools import accumulate, product, repeat
import numpy as np

def step_(z, c):
    z *= z
    z += c
    return z

def escapes(n, z, m, ws, cs, dtype=np.complex128):
    cs = (complex(x, y)
          for w, c in zip(ws, cs)
          for y, x in product(
              np.linspace(c.imag-w, c.imag+w, m),
              np.linspace(c.real-w, c.real+w, m)))
    cs = np.fromiter(cs, dtype=dtype, count=(len(ws)*m*m))
    zs = np.full_like(cs, z, dtype=dtype)
    zss = accumulate(repeat(cs, n), step_, initial=zs)
    ts = sum(np.abs(zs) < 2 for zs in zss)
    return ts.reshape(-1, m, m) # should add copy=False