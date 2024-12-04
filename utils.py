from pathlib import Path
from sys import float_info

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image


def zooms(l, z, w0, c0, cl):
    ws = np.geomspace(w0, w0/z, l, dtype=np.float64)
    rs = np.geomspace(1, 1/z, l, dtype=np.float64)
    qs = np.linspace(1, 0, l, dtype=np.float64)
    cs = (cl - (cl - c0) * rs * qs)
    return ws, cs

def extent(m, w, c):
    xs = np.linspace(c.real-w, c.real+w, m, dtype=np.float64)
    ys = np.linspace(c.imag-w, c.imag+w, m, dtype=np.float64)
    dx = (xs[-1] - xs[0]) / (len(xs) - 1) / 2
    dy = (ys[-1] - ys[0]) / (len(ys) - 1) / 2
    return (xs[0]-dx, xs[-1]+dx, ys[0]-dy, ys[-1]+dy)

def animate(frames, autoscale=False, imshow_kwargs=None, **kwargs):
    if imshow_kwargs is None: imshow_kwargs = {}
    fig, ax = plt.subplots()
    im = None
    def update(frame):
        nonlocal im
        m, w, c, data = frame
        if im is None:
            im = ax.imshow(data, extent=extent(m, w, c),
                           **imshow_kwargs)
            # im = ax.imshow(data)
        else:
            im.set_data(data)
            im.set_extent(extent(m, w, c))
            if autoscale: im.autoscale()
        return im,
    return FuncAnimation(fig, update, frames, **kwargs)

def save(arrs, autoscale=False, cmap=None, norm=None):
    _, ax = plt.subplots()
    im = None
    for i, arr in enumerate(arrs):
        if im is None:
            im = ax.imshow(arr, cmap=cmap, norm=norm)
        else:
            im.set_data(arr)
            if autoscale: im.autoscale()
        image = im.cmap(im.norm(arr))
        image = (image[:,:,:3] * 256).astype(np.uint8)
        file = f'./saveimgs/{i}.png'
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(file)
        plt.draw()
        plt.pause(float_info.epsilon)