from itertools import accumulate, product, repeat
import numpy as np
import torch

DEVICE = torch.device('mps')

def step_(z, c):
    z *= z
    z += c
    return z

def escapes(n, z, m, ws, cs, dtype=torch.complex64):
    cs = [complex(x, y)
          for w, c in zip(ws, cs)
          for y, x in product(
              np.linspace(c.imag-w, c.imag+w, m),
              np.linspace(c.real-w, c.real+w, m))]
    cs = torch.tensor(cs, dtype=dtype, device=DEVICE)
    zs = torch.full_like(cs, z, dtype=dtype, device=DEVICE)
    zss = accumulate(repeat(cs, n), step_, initial=zs)
    ms = sum(zs.abs() < 2 for zs in zss)
    return ms.cpu().view(-1, m, m)