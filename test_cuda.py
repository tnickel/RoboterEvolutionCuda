from numba import cuda
import numpy as np

@cuda.jit
def foo(x, y):
    i = cuda.grid(1)
    if i < x.size:
        y[i] = x[i] * 2

try:
    x = np.ones(10, dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    foo[1, 10](x, y)
    print("SUCCESS. Output:", y)
except Exception as e:
    print("FAILED:", e)
