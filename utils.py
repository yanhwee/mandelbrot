from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def zooms(l, z, w0, c0, cl):
    ws = np.geomspace(w0, w0/z, l)
    rs = np.geomspace(1, 1/z, l)
    qs = np.linspace(1, 0, l)
    cs = (cl - (cl - c0) * rs * qs)
    return ws, cs

def extent(m, w, c):
    d = w / (m - 1)
    return (c.real-d, c.real+d, c.imag-d, c.imag+d)

def animate(frames, autoscale=False, **kwargs):
    fig, ax = plt.subplots()
    im = None
    def update(frame):
        nonlocal im
        m, w, c, data = frame
        if im is None:
            im = ax.imshow(data, extent=extent(m, w, c))
        else:
            im.set_data(data)
            im.set_extent(extent(m, w, c))
        if autoscale: im.autoscale()
        return im,
    return FuncAnimation(fig, update, frames, **kwargs)