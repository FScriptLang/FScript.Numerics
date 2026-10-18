# Native BLAS and LAPACK

The backend uses the CBLAS and LAPACKE **C interfaces**, with LP64 (`int` dimensions/pivots). No Fortran ABI assumptions or public native pointers are exposed. `BlasSharp.OpenBlas` **0.3.0** supplies runtime assets transitively; the small adapter binds exported symbols directly. Its managed wrapper is not part of our public contract.

## Binary provenance and platforms

The [vendor NuGet package](https://www.nuget.org/packages/BlasSharp.OpenBlas/0.3.0) records [BlasSharp commit a8f916e](https://github.com/nuskey8/BlasSharp/tree/a8f916e848166ba4ace28c802e99d323a6534ba6). The macOS binary reports OpenBLAS **0.3.29.dev**, not a stable 0.3.29 build. We pin the package and every native binary's SHA-256 in [asset-sha256.json](native/asset-sha256.json); `make test` verifies the restored files. These hashes identify the actual shipped binaries independently of upstream version strings. We do not claim reproducible compilation of upstream assets from an identified OpenBLAS commit.

| RID | Asset included by vendor | Validation |
| --- | --- | --- |
| osx-arm64 | libopenblas.dylib | Native numerical, CLI/NuGet, LSP and DAP tests on development host |
| linux-arm64 | Mislabeled **x64** binary | Unsupported; rejected before loading |
| linux-x64 | libopenblas.so | x64 container validation and CI |
| win-x64 | libopenblas.dll | CI configured with GNU prerequisites; not locally executed |
| win-x86 | Mislabeled **x64** binary | Unsupported; rejected before loading |
| osx-x64 | Absent | Unsupported; despite vendor README listing macOS generally |

CI jobs must pass on their platform before a release claims that platform validated. Linux x64 requires glibc and the system Fortran runtime `libgfortran.so.5` (Debian/Ubuntu: `apt-get install libgfortran5`). This dependency is not bundled by the vendor NuGet; missing-runtime diagnostics retain the dynamic-loader cause. Linux assets do not target musl. Windows x64 requires the GNU x64 runtime DLLs `libgfortran-5.dll` and `libgcc_s_seh-1.dll` (plus their dependencies) on PATH, e.g. MSYS2 MinGW64 `mingw-w64-x86_64-gcc-libgfortran`; these are also absent from the vendor package. CI installs those prerequisites explicitly. Native ABI compatibility includes CPU dispatch and OS dependencies: the inspected macOS arm64 binary has dynamic architecture selection and only libSystem as a dylib dependency. It reports a maximum of three OpenBLAS threads. The linux-arm64 mismatch was confirmed by the ELF machine header and a failed ARM64 container load. Container/CI tests execute actual kernels rather than equating asset presence with support.

[OpenBLAS BSD license](native/OpenBLAS-LICENSE.txt) and [BlasSharp MIT license](native/BlasSharp-LICENSE.txt) are retained under `notices/` in our Blas package. See the [OpenBLAS project](https://github.com/OpenMathLib/OpenBLAS) for upstream architecture and licensing details.

## Loading and buffer policy

Loading is lazy: registration, inference, completion, hover, and metadata inspection do not initialize native code. First execution searches beside the Blas assembly, then `runtimes/<current-rid>/native`, then the normal .NET/assembly load-context resolver. This covers normal NuGet applications and FScript's staged package environment. A process-lifetime lazy handle retains delegates safely across simultaneous calls. It rejects `USE64BITINT` configurations and missing required float32/float64 CBLAS/LAPACKE symbols before numerical execution.

A missing/incompatible binary throws a descriptive initialization error naming the RID and package. The SDK translates it to an operation-named extension error with its inner cause. It never silently switches to a managed algorithm. Managed NDArray dot/matvec/matmul remain explicitly available.

The adapter borrows read-only spans for compatible contiguous inputs, including offsets. For matrices it recognizes C- and Fortran-contiguous layouts, computes transpose flags and leading dimensions, and pins the corresponding physical span. Unsupported positive-stride slices and zero-stride broadcast views are packed into temporary C-contiguous buffers. CBLAS only reads input pointers; callers must synchronize concurrent explicit mutations, just as for managed operations. BLAS float32 kernels accept all-Float32 operands; mixed floating inputs promote to Float64. LAPACK requires matching dtypes for right-hand sides. Bool/integer operands are rejected before native calls.

Buffers are pinned only for each synchronous call (`fixed` for BLAS; disposable GCHandles for LAPACK); all handles are released even on errors. Outputs use exclusive work buffers, then become independent NDArrays. LAPACK always copies its overwritten inputs. No native allocation ownership framework, permanent input pins, or collectible assembly contexts are introduced. No public writable span or borrowed native handle is exposed.

SDK `CheckRuntime` runs before and after each handler. An already-running synchronous native kernel cannot be interrupted safely; debugger pause/cancellation/limits take effect at the next host boundary. No native worker scheduler is added.

## Numerical validation and performance

Tests check float64 reconstruction at roughly 1e-10 and float32 at 2e-5 for small deterministic matrices. Larger or ill-conditioned problems need problem-specific tolerances, not bit equality. SVD-based rank decisions use dtype-aware relative cutoffs. NaNs/infinities are permitted by BLAS IEEE arithmetic; LAPACK and model training reject nonfinite inputs. Native errors are surfaced from LAPACKE `info` values.

`make benchmark` prints CSV timings and managed allocations for dot, gemv, gemm, contiguous/strided packing and reductions at dimensions 16, 128 and 512. The native path, including any required packing and fresh output allocation, is measured as used by consumers. Run separate processes with explicit thread settings:

```sh
OPENBLAS_NUM_THREADS=1 make benchmark
OPENBLAS_NUM_THREADS=2 make benchmark
```

OpenBLAS may read environment variables only during initialization. Limit its threads when running many FScript tasks, otherwise nested task/native parallelism can oversubscribe CPUs. Timings depend on hardware, native build and thread settings; the benchmark is a diagnostic, not a universal speed guarantee. See [recorded development results](benchmark-results.md).
