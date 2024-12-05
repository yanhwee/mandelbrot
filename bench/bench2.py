import matplotlib.pyplot as plt
import numpy as np

spacing = 0.2

width = (1 - spacing) / 2
labels = ['Pool.map(chunksize=200)\n(complex128)',
          'numpy (cpu)\n(complex128)',
          'numpy (cpu)\n(complex64)',
          'pytorch (gpu)\n(complex64)']
xs = np.arange(len(labels))
cpu = [764, 114, 122, 116]
real = [26.865, 12.881, 8.407, 3.793]
cpu_labels = [f'{c}%' for c in cpu]
real_labels = [f'{r}s' for r in real]

fig, ax1 = plt.subplots(layout='constrained')
ax2 = ax1.twinx()

rects1 = ax1.bar(xs + width * 0, cpu, width, color='orange')
rects2 = ax2.bar(xs + width * 1, real, width)

ax1.bar_label(rects1, cpu_labels, padding=3)
ax2.bar_label(rects2, real_labels, padding=3)

ax1.set_ylabel('cpu utilisation (%)')
ax2.set_ylabel('time taken (real) (secs)')
ax1.set_xticks(xs + width/2, labels)
ax1.legend([rects1, rects2], ['cpu', 'real'])

fig.suptitle('a very rough benchmark - 10 frames 200x200 max_iter=10k')
plt.show()