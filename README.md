# FScript.Numerics

Float64 numerical arrays for .NET and FScript. The independent C# library owns numerical storage; a separate extension exposes it through the FScript SDK.

| Package | Target | Purpose |
| --- | --- | --- |
| `FScript.Numerics` | .NET 10 | NDArrays, views, arithmetic, reductions, and matrix operations; no FScript dependency |
| `FScript.Numerics.Extensions` | .NET 10 | FScript provider, descriptors, and wrap/unwrap helpers; uses Extensibility 1.0.0 |

Both packages start at **0.1.0** and share a release version. The extension requires that exact version of the core. This repository prepares releases but does not publish packages during ordinary builds.

## Run an example

Install the .NET SDK pinned in `global.json`, then run from this repository:

```sh
make build
dotnet tool restore
dotnet tool run fscript -- examples/arrays.fss
dotnet tool run fscript -- examples/matrices.fss
dotnet tool run fscript -- examples/views.fss
```

The local tool manifest pins the published FScript CLI to **0.82.0**. Each example uses the adjacent [`fscript.toml`](examples/fscript.toml), which points to the built provider and discovers it automatically. With a global FScript 0.82.0 installation, `fscript examples/arrays.fss` also works. Opening this folder in VS Code discovers the same manifest; select it with **FScript: Select Configuration** if another configuration is selected. Build the provider first and trust the workspace to enable execution. Windows may require stopping the language server before rebuilding a DLL it has loaded.

```fsharp
let values = NDArray.ofList [1.0; 2.0; 3.0]
let doubled = NDArray.multiplyScalar values 2.0
(NDArray.toList values, NDArray.toList doubled, NDArray.sum doubled)
// ([1;2;3], [2;4;6], 12)
```

FScript bindings and visible fields remain immutable. `NDArray.set` explicitly mutates hidden numerical storage; shared views observe the change. Arithmetic and `copy` allocate independent storage. Ordinary `=` compares array identity; `NDArray.equals` and `NDArray.allClose` compare numerical contents.

## Use from .NET

```csharp
using FScript.Numerics;

var a = NDArray.FromRows([1, 2], [3, 4]);
var view = a.Transpose();
var copy = a.Copy();
view.Set(9, 1, 0);                 // a.Get(0, 1) is now 9
var product = a.Matmul(copy);      // new storage
```

Constructors copy inputs. Shape and stride metadata are read-only, and strides count **elements**, not bytes. The core uses managed `double[]` storage and System.Numerics.Tensors 10.0.12 where contiguous kernels apply.

## Documentation and development

- [FScript function reference and embedding](docs/specs/fscript-api.md)
- [.NET API and numerical behavior](docs/specs/ndarray.md)
- [Memory layout, mutation, and architecture](docs/architecture/overview.md)
- [Contribution, validation, and release process](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

```sh
make build             # warning-free Release build
make test              # numerical, SDK/runtime integration, and release-tool tests
make smoke-tests       # every example through the published CLI
make pack              # two packages + symbols, verified dependency graphs
make verify-changelog
make release-prepare version=0.1.0 dryrun=true  # requires a clean committed tree
```

Only float64 is included. General broadcasting, axis reductions, negative indexing/slicing, additional dtypes, native storage, BLAS/LAPACK, and in-place arithmetic are outside this release. Explicit element mutation is supported through `Set`/`set`. Concurrent reads and writes require caller synchronization.
