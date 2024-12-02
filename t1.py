from itertools import repeat

from matplotlib import pyplot as plt
import numpy as np
from utils import animate, zooms
import vanilla
import vnumba
import vnumpy
import vpytorch

if __name__ == '__main__':
    l = 10 # frames
    m = 200 # pixels
    n = 10000 # escapes
    w0 = 1.5 # window size
    x0, y0 = -1, 0
    x1 = -0.74453986035590838011
    y1 = 0.12172377389442482241
    z = int(1e6) # zoom
    c0, cl = complex(x0, y0), complex(x1, y1)
    ws, cs = zooms(l, z, w0, c0, cl)
    # arrs = vanilla.escapes_(n, 0, m, ws, cs)
    # arrs = vanilla.escapes(n, 0, m, ws, cs)
    # arrs = vnumba.escapes(n, 0, m, ws, cs)
    # arrs = vnumpy.escapes(n, 0, m, ws, cs, dtype=np.complex64)
    arrs = vpytorch.escapes(n, 0, m, ws, cs)
    frames = zip(repeat(m), ws, cs, arrs)
    ani = animate(frames, autoscale=True,
                  save_count=l, interval=1000,
                  repeat=True)
    plt.show()
    ani.save('./images/t1.png', writer='pillow')