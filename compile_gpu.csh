#!/bin/tcsh -f
# Compile sfcalc_gpu.cu into a shared library on voltron
# Run this on voltron (not via srun -- compilation is CPU-only)

source /programs/cuda/setup_cuda.csh

cd /home/jamesh/projects/fft_symmetry/claude_test

echo "Compiling sfcalc_gpu.cu ..."
nvcc -O3 -arch=sm_70 \
     -shared -Xcompiler -fPIC \
     -lcufft \
     sfcalc_gpu.cu \
     -o sfcalc_gpu.so

if ( $status == 0 ) then
    echo "OK: sfcalc_gpu.so"
    ls -lh sfcalc_gpu.so
else
    echo "FAILED"
    exit 1
endif
