from itertools import batched, repeat
from math import ceil
from operator import itemgetter
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import animate, save

DATA_FILE = './data/data1.npz'
COORDS_FILE = './data/coords1.npz'

def load():
    with np.load(DATA_FILE, mmap_mode='r') as file:
        arrs, = file.values()
    with np.load(COORDS_FILE, mmap_mode='r') as file:
        ws, cs = file.values()
    return arrs, ws, cs

if __name__ == '__main__':
    arrs, ws, cs = load()
    l, m, m = arrs.shape
    frames = zip(repeat(m), ws, cs, arrs)
    frames = tqdm(frames, total=l)
    frames = map(itemgetter(0), batched(frames, 8))
    # arrs = map(itemgetter(3), frames)
    ani = animate(
        frames, autoscale=True, save_count=l,
        interval=4, repeat=True,
        cache_frame_data=False,
        imshow_kwargs={'norm': 'log', 'cmap': 'magma'})
    plt.show()
    # print('Saving as image...')
    # ani.save('./images/render19.png', writer='pillow')
    # save(arrs, autoscale=True, cmap='magma', norm='log')