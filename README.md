# FScript.Numerics

Numerical arrays, native linear algebra, statistics, and classical machine learning for .NET 11 and FScript. The numerical libraries are independent of FScript; one SDK extension supplies the script environment.

| Package | Purpose | Dependencies within this repository |
| --- | --- | --- |
| `FScript.Numerics` | Five dtypes, shared views, broadcasting, arithmetic, reductions | None |
| `FScript.Numerics.Blas` | OpenBLAS dot, norm, matrix-vector and matrix multiplication | Numerics |
| `FScript.Numerics.Lapack` | Solvers, LU, QR, Cholesky, SVD, symmetric eigen | Numerics, Blas |
| `FScript.Numerics.Statistics` | Descriptive statistics and seeded distributions | Numerics |
| `FScript.Numerics.ML` | Scalers, regression, clustering, PCA, metrics | Numerics, Lapack, Statistics |
| `FScript.Numerics.Extensions` | FScript descriptors, monomorphic schemes, opaque wrappers | All five libraries; Extensibility 2.0.0 |

All packages share version **0.1.0** and require the matching companion versions. The core and numerical modules have no FScript or FSharp.Core dependency. The production extension has no Language or Runtime dependency.

## Start with arrays

```fsharp
let a = NDArray.ofRows [[1.0;2.0;3.0];[4.0;5.0;6.0]]
let b = NDArray.add a (NDArray.ofList [10.0;20.0;30.0])
NDArray.toRows b
// [[11;22;33];[14;25;36]]
```

Bindings and visible fields are immutable. **Explicit `NDArray.set` mutates storage** and ordinary views share that storage. Arithmetic, `copy`, native operations, and model fitting preserve their inputs. Broadcast views reject mutation. Ordinary `=` compares per-instance identity; `NDArray.equals` compares contents; `NDArray.equal` returns an elementwise bool mask.

## Run the examples

With the pinned .NET SDK installed:

```sh
make build
dotnet tool restore
dotnet tool run fscript -- examples/broadcasting.fss
dotnet tool run fscript -- examples/linear-algebra.fss
dotnet tool run fscript -- examples/statistics.fss
dotnet tool run fscript -- examples/regression.fss
dotnet tool run fscript -- examples/kmeans.fss
dotnet tool run fscript -- examples/pca.fss
```

The tool manifest pins published FScript **0.94.3** for integration. A global installation also works: `fscript examples/pca.fss`. The adjacent [`fscript.toml`](examples/fscript.toml) selects the built provider. VS Code discovers that same file; use **FScript: Select Configuration** if needed. Trust the workspace, build first, and use **FScript: Debug Current Script** to inspect metadata. Large numerical storage stays hidden. Windows may require stopping the server before rebuilding a DLL it has loaded.

After packages are published, consumers can replace local DLL paths with:

```toml
[[runtime.packages]]
name = "FScript.Numerics.Extensions"
version = "0.1.0"
```

The extension aggregates all modules and registers NDArray once. FScript restores one dependency graph into the manifest-local `.fscript` cache. [Compatibility guidance](docs/compatibility.md) explains assembly identity and embedded hosts.

## Use from .NET

```csharp
using FScript.Numerics;
using FScript.Numerics.Blas;
using FScript.Numerics.Lapack;

var a = NDArray.FromRows([3, 1], [1, 2]);
var b = NDArray.FromArray([9, 8]);
var solution = Decompositions.Solve(a, b); // [2, 3]
var product = LinearAlgebra.Matmul(a, a); // fresh storage
```

Native functions require float32/float64. OpenBLAS loads only on first numerical use, with LP64 validation and actionable errors. See the [native support matrix](docs/native-backends.md) before deployment.

## Documentation and development

- [NDArray contract and .NET API](docs/specs/ndarray.md)
- [Every FScript signature and descriptor field](docs/specs/registration.md)
- [Operation arguments and numerical policies](docs/specs/operations.md)
- [Embedding and SDK ownership](docs/specs/fscript-api.md)
- [Architecture and mutation diagrams](docs/architecture/overview.md)
- [Native assets, licensing, threads, and benchmarks](docs/native-backends.md)
- [Grounding and milestone findings](docs/expansion-grounding.md)
- [Validation record and platform boundaries](docs/validation.md)
- [Contribution and release process](CONTRIBUTING.md)

```sh
make build test smoke-tests pack verify-changelog
make package-smoke-tests  # local NuGet graph -> published CLI/server/debugger
make benchmark           # timings and allocations; run with chosen BLAS thread settings
make reference           # regenerate the checked SDK reference
make release-prepare version=0.1.0 dryrun=true # requires a clean committed tree
```

Negative indexing/slicing, sparse/GPU arrays, general batched algebra, in-place arithmetic, pipelines, Data and Plotting are outside this change. Concurrent writes require caller synchronization.

## Compiler and concurrency contracts

The extension declares [purity and thread-safety metadata](docs/specs/fscript-api.md#compiler-purity-and-task-sharing-contracts). Mutable arrays are neither pure nor promised safe for concurrent sharing. Immutable fitted objects have type guarantees; factories with only immutable inputs may be thread-safe while remaining non-pure. Numerical validation failures are preserved.

For compiled .NET performance, the optional [FScript/F# array comparison](benchmarks/FScript.Numerics.CompilerComparison/README.md) measures real SDK FFI against direct F# NDArray calls and ordinary F# arrays. It requires a local FScript source checkout; regular builds and packages do not.
