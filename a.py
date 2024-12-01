from collections import deque
from itertools import accumulate, count, repeat, takewhile
from operator import mul, sub

from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation, FuncAnimation
import numpy as np
from tqdm import tqdm

def consume(xs):
    deque(xs, maxlen=0)

def ilen(xs):
    counter = count()
    consume(zip(xs, counter))
    return next(counter)

def step(z, c):
    return z * z + c

def escape(n, z, c):
    seq = accumulate(repeat(c, n), step, initial=z)
    seq = takewhile(lambda z: abs(z) < 2, seq)
    return ilen(seq)

def escapes(n, z, xs, ys):
    return [[escape(n, z, complex(x, y))
             for x in xs] for y in ys]

def zooms(m, z, n, w0, x0, y0, xm, ym):
    z0, zm = np.array([x0, y0]), np.array([xm, ym])
    dz = zm - z0
    r = (1 / z) ** (1 / m)
    ws = accumulate(repeat(r, m), mul, initial=w0)
    rs = accumulate(repeat(r, m), mul, initial=1)
    qs = accumulate(repeat(1/m, m), sub, initial=1)
    zs = (zm - dz * r * q for r, q in zip(rs, qs))
    return (np.linspace(-w, w, n) + z.reshape(2, 1)
            for w, z in zip(ws, zs))

def extent(xs, ys):
    dx = (xs[0] - xs[-1]) / (len(xs) - 1) / 2
    dy = (ys[0] - ys[-1]) / (len(ys) - 1) / 2
    return (xs[0]-dx, xs[-1]+dx, ys[0]-dy, ys[-1]+dy)

def animate(frames):
    fig, ax = plt.subplots()
    im = ax.imshow([[]])
    def update(frame):
        xs, ys, data = frame
        im.set_data(data)
        im.set_extent(extent(xs, ys))
        return im,
    ani = FuncAnimation(
        fig=fig, func=update, frames=frames,
        interval=10)
    plt.show()

if __name__ == '__main__':
    m, z = 100, int(1e18)
    n, w0 = 200, 1.5
    x0, y0 = -1, 0
    x1, y1 = 0, 1
    xsys = zooms(m, z, n, w0, x0, y0, x1, y1)
    fig, ax = plt.subplots()
    artists = ([ax.imshow(escapes(200, 0, xs, ys))]
               for xs, ys in xsys)
    artists = list(tqdm(artists, total=m+1))
    ani = ArtistAnimation(fig=fig, artists=artists,
                          interval=100, repeat=False)
    plt.show()