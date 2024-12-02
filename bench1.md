```
l = 10 # frames
m = 200 # pixels
n = 10000 # escapes
w0 = 1.5 # window size
x0, y0 = -1, 0
x1 = -0.74453986035590838011
y1 = 0.12172377389442482241
z = int(1e6) # zoom
```

(vanilla._escapes) python t1.py  114.73s user 0.35s system 101% cpu 1:53.25 total

(vanilla.escapes map) python t1.py  192.03s user 2.91s system 583% cpu 33.417 total

(vanilla.escapes imap) python t1.py  217.46s user 11.57s system 713% cpu 32.113 total

(vanilla.escapes imap chunksize=m) python t1.py  200.66s user 2.17s system 764% cpu 26.527 total

(vanilla.escapes map chunksize=m) python t1.py  203.53s user 1.94s system 764% cpu 26.875 total

(vnumba.escapes) python t1.py  6.17s user 0.52s system 138% cpu 4.838 total

(vnumba.escapes disabled) python t1.py  87.82s user 0.39s system 101% cpu 1:26.91 total

(vnumpy.escapes complex128) python t1.py  13.74s user 1.02s system 114% cpu 12.881 total

(vnumpy.escapes complex64) python t1.py  9.77s user 0.55s system 122% cpu 8.407 total

(vpytorch.escapes complex64) python t1.py  3.76s user 0.65s system 116% cpu 3.793 total

