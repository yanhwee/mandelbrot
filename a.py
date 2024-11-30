from collections import deque
from itertools import accumulate, count, repeat, takewhile

from matplotlib import pyplot as plt
import numpy as np

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

if __name__ == '__main__':
    x, y = -1, 0
    dw, n = 1.5, 100

    ws = np.linspace(-dw, dw, num=n)
    hp = dw / (n - 1)
    xs, ys = ws + x, ws + y
    extent = (xs[0]-hp, xs[-1]+hp, ys[0]-hp, ys[-1]+hp)
    
    arr = escapes(100, 0, xs, ys)
    plt.imshow(arr, extent=extent)
    plt.show()