from matplotlib import pyplot as plt
import numpy as np

MAX_ITERATION = 100

def escape(c, z=complex(0, 0)):
    i = 0
    while abs(z) < 2 and i < MAX_ITERATION:
        z = z * z + c
        i += 1
    return i

def image(xs, ys):
    return [[escape(complex(x, y))
             for x in xs] for y in ys]

if __name__ == '__main__':
    im = image(np.linspace(-2.5,0.5,100),
               np.linspace(-1.5,1.5,100))
    plt.imshow(im)
    plt.show()