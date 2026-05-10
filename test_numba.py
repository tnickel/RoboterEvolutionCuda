import os
import sys

# Finde nvidia pip packages
import nvidia.cuda_nvcc
import nvidia.cuda_runtime
import nvidia.cuda_nvrtc

nvcc_path = os.path.dirname(nvidia.cuda_nvcc.__file__)
os.environ['NUMBA_CUDA_TOOLKIT'] = nvcc_path

from numba import cuda
print("CUDA Available:", cuda.is_available())
print("CUDA Detect:")
cuda.detect()
