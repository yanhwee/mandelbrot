from collections import deque
from functools import partial
from itertools import accumulate, batched, count, product, repeat, takewhile
from multiprocessing import Pool

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

def escapes(n, z, m, ws, cs):
    cs = (complex(x, y)
          for w, c in zip(ws, cs)
          for y, x in product(
              np.linspace(c.imag-w, c.imag+w, m),
              np.linspace(c.real-w, c.real+w, m)))
    with Pool() as p:
        ts = p.imap(partial(escape, n, z), cs, chunksize=m)
        return list(batched(batched(ts, m), m))

def escapes_(n, z, m, ws, cs):
    xsys = ((
        np.linspace(c.real-w, c.real+w, m),
        np.linspace(c.imag-w, c.imag+w, m))
        for w, c in zip(ws, cs))
    return [[[escape(n, z, complex(x, y))
              for x in xs] for y in ys]
              for xs, ys in xsys]
