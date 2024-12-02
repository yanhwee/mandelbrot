from functools import partial
from itertools import accumulate, repeat, starmap
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from a import animate, zooms

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
    m = 10 # frames
    n = 100 # pixels
    k = 10000 # escapes
    w0 = 1.5 # window size
    x0, y0 = -1, 0
    x1 = -0.74453986035590838011
    y1 = 0.12172377389442482241
    # z = int(1e15) # zoom
    z = int(1e6)
    xsys = zooms(m-1, z, n, w0, x0, y0, x1, y1)
    xsys = list(xsys)
    arrs = starmap(partial(escapes, k, 0), tqdm(xsys))
    frames = ((xs, ys, arr) for (xs, ys), arr in zip(xsys, arrs))
    ani = animate(frames, autoscale=True,
                  save_count=m, interval=1000,
                  repeat=True)
    plt.show()
    ani.save('./images/img-gpu.png', writer='pillow')
    