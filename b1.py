from itertools import accumulate, repeat
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

DEVICE = torch.device('mps')

def step_(z, c):
    z *= z
    z += c
    return z

def escapes_(n, z, xs, ys, dtype=torch.complex64):
    cs = [complex(x, y) for y in ys for x in xs]
    cs = torch.tensor(cs, dtype=dtype, device=DEVICE)
    zs = torch.full_like(cs, z, dtype=dtype, device=DEVICE)
    zss = accumulate(repeat(cs, n), step_, initial=zs)
    return (zs.abs().cpu().reshape(len(ys), len(xs))
            for zs in zss)

if __name__ == '__main__':
    x, y = -1, 0
    dw, n = 1.5, 100

    ws = np.linspace(-dw, dw, num=1000)
    hp = dw / (n - 1)
    xs, ys = ws + x, ws + y
    extent = (xs[0]-hp, xs[-1]+hp, ys[0]-hp, ys[-1]+hp)

    l = 65
    arrs = escapes_(l-1, 0, xs, ys)
    frames = ((i, arr + 1) for i, arr in enumerate(arrs))
    fig, ax = plt.subplots()
    im = None
    def update(frame):
        global im
        i, arr = frame
        if im is None:
            im = ax.imshow(arr, extent=extent,
                           norm='log',
                           vmin=1,
                           vmax=1e5)
            fig.colorbar(im, ax=ax)
        else:
            im.set_data(arr)
            # im.autoscale()
        fig.suptitle(f'magnitude (iteration {i})')
    ani = FuncAnimation(fig, update, frames, save_count=l)
    plt.show()
    # ani.save('animation.mp4', 'ffmpeg', fps=2)