# Development benchmark results

Recorded on macOS arm64 with .NET SDK 10.0.401 and the pinned OpenBLAS 0.3.29.dev asset. Two fresh processes used OPENBLAS_NUM_THREADS=1 and 2. These short diagnostics include JIT/GC noise and concurrent system activity; they are not statistically rigorous regression thresholds. Reproduce with `make benchmark` and collect repeated runs on deployment hardware.

Matrix size is the side length; dot size is vector length. Native calls include layout adaptation and fresh result allocation. C/F inputs are borrowed read-only; strided copies are measured separately. Managed bytes are measured on the calling thread and exclude native workspaces and native worker allocations. Large GEMM currently includes both a private output buffer and the defensive NDArray constructor copy.

| Operation | Size | 1 thread, ms/call | 2 threads, ms/call | Managed bytes/call (1 thread) |
| --- | ---: | ---: | ---: | ---: |
| dot/native | 16 | 0.000600 | 0.000525 | 112 |
| gemv/native | 16 | 0.001055 | 0.001180 | 664 |
| gemm/native | 16 | 0.001560 | 0.001585 | 4504 |
| gemm/transposed | 16 | 0.003195 | 0.003130 | 6344 |
| copy/contiguous | 16 | 0.000170 | 0.000200 | 2080 |
| copy/strided | 16 | 0.014815 | 0.014535 | 3080 |
| sum/contiguous | 16 | 0.000810 | 0.000760 | 8 |
| sum/strided | 16 | 0.017680 | 0.018225 | 1008 |
| dot/native | 128 | 0.000605 | 0.000575 | 112 |
| gemv/native | 128 | 0.003290 | 0.003385 | 2456 |
| gemm/native | 128 | 0.120070 | 0.087860 | 262552 |
| gemm/transposed | 128 | 0.122625 | 0.095975 | 264392 |
| copy/contiguous | 128 | 0.007275 | 0.007460 | 131104 |
| copy/strided | 128 | 0.636725 | 0.628450 | 132104 |
| sum/contiguous | 128 | 0.084800 | 0.087755 | 8 |
| sum/strided | 128 | 0.604475 | 0.589975 | 1008 |
| dot/native | 512 | 0.000967 | 0.000900 | 160 |
| gemv/native | 512 | 0.036333 | 0.037467 | 8648 |
| gemm/native | 512 | 5.397600 | 3.258533 | 4194760 |
| gemm/transposed | 512 | 5.456900 | 3.276833 | 4199312 |
| copy/contiguous | 512 | 0.207033 | 0.211800 | 2097232 |
| copy/strided | 512 | 1.764167 | 9.396167 | 2100938 |
| sum/contiguous | 512 | 0.125500 | 0.130833 | 56 |
| sum/strided | 512 | 1.528667 | 1.505400 | 1056 |
| vector-dot/native | 256 | 0.000950 | 0.000920 | 120 |
| vector-dot/managed | 256 | 0.000860 | 0.000790 | 16 |
| vector-dot/native | 16384 | 0.003520 | 0.002170 | 120 |
| vector-dot/managed | 16384 | 0.131120 | 0.129450 | 16 |
| vector-dot/native | 1048576 | 0.158250 | 0.079290 | 120 |
| vector-dot/managed | 1048576 | 0.449820 | 0.446910 | 16 |

Contiguous/transpose-aware BLAS paths avoid input-sized temporary allocations. Unsupported strided layouts incur explicit packing costs. Small vectors may favor the managed TensorPrimitives path; large BLAS kernels and thread settings must be measured for the workload. The benchmark exposes both alternatives rather than assuming every native call is faster.
