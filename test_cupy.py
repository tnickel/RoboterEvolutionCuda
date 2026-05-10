import cupy as cp

kernel_code = '''
extern "C" __global__
void test(float* x) {
    int i = threadIdx.x;
    x[i] *= 2.0;
}
'''
kernel = cp.RawKernel(kernel_code, 'test')
x = cp.array([1.0, 2.0], dtype=cp.float32)
kernel((1,), (2,), (x,))
print(x)
