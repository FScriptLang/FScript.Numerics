# Numerics expansion: grounded findings

## Baseline before expansion

At the start of the audit, the repository contained two .NET 10 C# libraries, both version 0.1.0:
`FScript.Numerics` and `FScript.Numerics.Extensions`. The numerical core has no
FScript or FSharp.Core dependency. The extension references the published
MagnusOpera.FScript.Extensibility 1.0.0; only its test project references Runtime.

The baseline warning-free build passes 13 core tests, 13 extension tests and five
release-tool tests. Existing factories include scalars, vectors, rectangular rows,
zeros/ones/full, stop-exclusive arange, and identity. Layout uses owner-anchored
managed double[] storage and nonnegative **element strides**. Scalars and empty
shapes are implemented. Transpose, permutation, slicing and contiguous reshape
share storage; copy isolates it. Noncontiguous reshape currently rejects input.

The baseline arithmetic required exact shapes. There are whole-array sum/mean/min/max
reductions and managed dot/matvec/matmul. DType contains Float64 only. Axis
reductions, broadcasting, composition, conversions, native BLAS/LAPACK,
statistics and ML are absent. Preserve the existing array-first FScript arguments.

The current `Set` / `NDArray.set` contract deliberately mutates shared storage.
The proposal’s blanket immutability rule conflicts with this existing contract.
The user explicitly confirmed retaining mutation; that decision governs this
implementation.

## Existing extension and host contracts

The provider registers one opaque NDArray type with Id, Rank, Size, Shape and
DType snapshots. A weak-key identity table preserves the same Id when wrapping
the same object. Ordinary FScript equality compares visible snapshots; equals
and allClose compare contents. Host-managed construction/updates are rejected by
the interpreter. The opaque payload remains the concrete numerical NDArray.

SDK descriptors and schemes already support additional opaque fitted-model
values and structured results; no language or SDK changes are needed. Keep
Language and Runtime dependencies confined to integration tests.

FScript's NuGet loader restores one dependency graph per environment, rejects
conflicting assembly files with the same simple name, loads shared numerical
assemblies once in that graph, and shares SDK and FSharp.Core with the host.
Native assets are staged both beside managed assemblies and under their RID
paths. Independent `extern-assemblies` loading instead uses normal .NET assembly
loading: integration tests must cover both paths and prevent duplicate NDArray
registrations when modules depend on the base extension.

The debugger expands host-managed visible fields only; it does not reflect the
NDArray payload or call ToArray. Keeping storage hidden and metadata bounded
therefore needs no new debugger feature. SDK CheckRuntime is available before
and after host calls. An in-flight synchronous native kernel cannot be safely
interrupted; do not claim otherwise or add a second scheduling convention.

## Native backend investigation

Candidate: pinned BlasSharp.OpenBlas 0.3.0, whose NuGet repository metadata points
to nuskey8/BlasSharp commit a8f916e848166ba4ace28c802e99d323a6534ba6.
Primary sources: https://www.nuget.org/packages/BlasSharp.OpenBlas/0.3.0 and
https://github.com/nuskey8/BlasSharp .

Inspection of the actual package finds linux-x64, linux-arm64, osx-arm64, win-x64
and win-x86 native assets. **There is no osx-x64 binary**, despite the README's
platform list. Do not claim that RID. Package presence alone is not validation.

The osx-arm64 binary loads on the current host, exports float32/float64 CBLAS and
LAPACKE entry points, and reports:
`OpenBLAS 0.3.29.dev DYNAMIC_ARCH NO_AFFINITY neoversen1 MAX_THREADS=3`.
Its only dependent dylib is the system libSystem; its install name contains an
upstream build-runner path. It does not report USE64BITINT. Before adopting it,
verify LP64 at runtime, test actual kernels, record per-RID binary hashes, retain
upstream license notices, and test loading from restored packages and CLI hosts.
Native loading must be lazy so missing libraries do not break type discovery.

## Confirmed delivery decisions

- Keep explicit Set/set mutation, including shared ordinary views. Broadcast
  views remain read-only. Arithmetic, native calls and fitted-model operations
  preserve existing inputs. This user clarification supersedes the proposal’s
  blanket immutability language.
- Implement M0–M5 in this change, without publishing packages automatically.
- Keep the numerical projects independent of FScript. The existing Extensions
  project will aggregate SDK registrations for all modules, preventing duplicate
  NDArray descriptors and reusing the same concrete numerical assembly.
- Implement Float64, Float32, Int32, Int64 and Bool storage, explicit conversions
  and broadcasting before adding native modules. BLAS/LAPACK accept floating
  arrays only. Preserve array-first FScript argument conventions.

The audit is complete. Implementation and validation milestones follow; each includes tests and an executable example.

- [x] M0 baseline and native ABI probe (ddot, sgemm, dgemm, LAPACKE_dgesv).
- [x] M1 dtypes, broadcasting, views, reductions, composition, SDK API.
- [x] M2 packaged native BLAS, diagnostics, benchmarks and interop.
- [x] M3 native LAPACK decompositions and solvers.
- [x] M4 descriptive statistics and seeded distributions.
- [x] M5 fitted models, metrics and complete integration validation.

## Final native findings and validation boundaries

The vendor linux-arm64 ELF header is x86-64, and its win-x86 PE header is also
x86-64. Native execution rejects both mislabeled targets explicitly. There is no
macOS x64 asset. Linux x64 additionally needs the system libgfortran.so.5;
Windows x64 needs GNU Fortran/GCC runtime DLLs on PATH. These prerequisites are
not bundled by the vendor. Hash/machine checks and the documented RID matrix
record actual asset contents rather than trusting package labels.

Native unit/integration tests and CLI examples pass on the macOS arm64 host and
Linux x64 container (with libgfortran5). Restored local NuGet artifacts pass
published CLI, LSP and debugger tests on macOS arm64 and Linux x64. Windows validation is wired
into CI with explicit GNU prerequisites, but has not been executed locally.
BLAS borrows compatible C/F inputs and packs unsupported layouts; LAPACK copies
overwritten inputs. Benchmark diagnostics record packing and allocation costs. No FScript, SDK, Terrabuild or umbrella files changed.
