from itertools import accumulate, repeat
from matplotlib import pyplot as plt
import numpy as np
import torch

DEVICE = torch.device('mps')

def step_(z, c):
    z *= z
    z += c
    return z

def escapes(n, z, xs, ys, dtype=torch.complex64):
    cs = [complex(x, y) for y in ys for x in xs]
    cs = torch.tensor(cs, dtype=dtype, device=DEVICE)
    zs = torch.full_like(cs, z, dtype=dtype, device=DEVICE)
    zss = accumulate(repeat(cs, n), step_, initial=zs)
    ms = sum(zs.abs() < 2 for zs in zss)
    return ms.cpu().reshape(len(ys), len(xs))

if __name__ == '__main__':
    x, y = -1, 0
    dw, n = 1.5, 1000

    ws = np.linspace(-dw, dw, num=n)
    hp = dw / (n - 1)
    xs, ys = ws + x, ws + y
    extent = (xs[0]-hp, xs[-1]+hp, ys[0]-hp, ys[-1]+hp)

    arr = escapes(100, 0, xs, ys)
    plt.imshow(arr, extent=extent)
    plt.show()