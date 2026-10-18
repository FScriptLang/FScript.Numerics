# FScript API and embedding

Build the provider with `make build`. FScript CLI 0.82.0 automatically discovers the manifest beside an example script:

```sh
fscript examples/views.fss
# Or use the repository's pinned local tool:
dotnet tool restore
dotnet tool run fscript -- examples/views.fss
```

The manifest contains:

```toml
[runtime]
extern-assemblies = ["../src/FScript.Numerics.Extensions/bin/Release/net10.0/FScript.Numerics.Extensions.dll"]
```

Paths resolve against the manifest directory. VS Code uses this same configuration for completion, hover, and diagnostics; build first and allow provider execution only in a trusted workspace. No additional extension configuration is required. Users with several manifests can select this one through **FScript: Select Configuration**. Runtime and CLI 0.82.0 are pinned for this repository's integration tests; they are not production dependencies of the provider.

## NDArray descriptor

`NumericsExtension.CreateRegistration()` returns an SDK `ExtensionRegistration` containing one type, no unions, and 36 functions. `NumericsProvider.Provide()` is a zero-argument `[FScriptExternProvider]` entry point returning the same registration.

| Descriptor field | Value | Purpose |
| --- | --- | --- |
| `Name` | `NDArray` | Nominal script type, used in every array-taking or array-returning scheme |
| `Fields` | The map below | Immutable visible metadata; never numerical storage |
| `HostManaged` | `true` | Script construction and record updates forbidden |

| Visible field | Script type | Purpose |
| --- | --- | --- |
| `Id` | `string` | Stable identity for this actual core NDArray instance |
| `Rank` | `int` | Number of dimensions |
| `Size` | `int` | Logical element count |
| `Shape` | `int list` | Immutable dimension snapshot |
| `DType` | `string` | Always `"float64"` |

`NDArray` is opaque: numerical storage lives in its payload. Field access and destructuring are supported, but script literals and both record-update syntaxes are rejected. `Id` distinguishes views and independent copies, even if the values are equal. Rewrapping the same instance preserves its Id through weak-key bookkeeping.

## Function descriptors

All names below have prefix `NDArray.`. Every scheme is monomorphic: the quantified variable list is empty. Calls are curried, and array-taking operations put the array first. Function descriptors are authored directly in C# against the SDK, so the independent numerical core requires no FScript annotations.

| `ExternalFunction` field | Purpose |
| --- | --- |
| `Name` | Fully qualified name, such as `NDArray.get` |
| `Scheme` | Authoritative `FunctionScheme.Forall([], ScriptType)` encoding the signature below |
| `Arity` | Number of curried arguments consumed before invocation |
| `Impl` | SDK invocation-context/list handler; validates arguments/results and runtime limits |

### Construction

| Function | Signature | Behavior |
| --- | --- | --- |
| `scalar` | `float -> NDArray` | Rank-zero single element |
| `ofList` | `float list -> NDArray` | Copied vector |
| `ofRows` | `float list list -> NDArray` | Copied rectangular matrix |
| `zeros` | `int list -> NDArray` | Shape filled with zero |
| `ones` | `int list -> NDArray` | Shape filled with one |
| `full` | `int list -> float -> NDArray` | Shape filled with value |
| `arange` | `float -> float -> float -> NDArray` | Start, stop-exclusive, step |
| `identity` | `int -> NDArray` | Square identity matrix |

Use float literals such as `1.0`; integer values are not coerced to floats. Shape dimensions and indices are checked conversions from script int64 to .NET int32. Empty `ofRows []` has shape `[0;0]`, while `ofRows [[];[]]` has shape `[2;0]`. `zeros []` is a scalar zero. `arange` requires finite inputs and nonzero step; incompatible direction produces an empty vector.

### Access and conversion

| Function | Signature | Behavior |
| --- | --- | --- |
| `get` | `NDArray -> int list -> float` | Exactly one nonnegative in-bounds index per axis |
| `set` | `NDArray -> int list -> float -> unit` | Explicit one-element mutation; visible fields stay immutable |
| `toList` | `NDArray -> float list` | Independent flattened logical row-major values, any rank |
| `toRows` | `NDArray -> float list list` | Independent row lists; rank two only |

### Views and copies

| Function | Signature | Behavior |
| --- | --- | --- |
| `transpose` | `NDArray -> NDArray` | Shared view with all axes reversed |
| `permute` | `NDArray -> int list -> NDArray` | Shared view with explicit complete axis permutation |
| `reshape` | `NDArray -> int list -> NDArray` | Shared contiguous view with unchanged element count |
| `slice` | `NDArray -> int -> int -> int -> int -> NDArray` | Array, axis, start, stop-exclusive, positive step |
| `copy` | `NDArray -> NDArray` | Independent contiguous copy |
| `contiguous` | `NDArray -> NDArray` | Same instance when contiguous, otherwise copy |

Slices preserve rank and require `0 <= start <= stop <= dimension`. There is no negative indexing or implicit clipping. Non-contiguous reshape fails; explicitly copy first. Singleton strides do not affect contiguity; offset contiguous views and empty layouts are supported.

### Arithmetic

| Function | Signature | Behavior |
| --- | --- | --- |
| `add` | `NDArray -> NDArray -> NDArray` | Elementwise addition |
| `subtract` | `NDArray -> NDArray -> NDArray` | Elementwise subtraction |
| `multiply` | `NDArray -> NDArray -> NDArray` | Elementwise multiplication |
| `divide` | `NDArray -> NDArray -> NDArray` | Elementwise division |
| `addScalar` | `NDArray -> float -> NDArray` | Add a right-hand scalar |
| `subtractScalar` | `NDArray -> float -> NDArray` | Subtract a right-hand scalar |
| `multiplyScalar` | `NDArray -> float -> NDArray` | Multiply by a right-hand scalar |
| `divideScalar` | `NDArray -> float -> NDArray` | Divide by a right-hand scalar |
| `negate` | `NDArray -> NDArray` | Unary negation |

Every operation produces new storage and leaves inputs unchanged. Array shapes must match exactly; rank-zero array operands do not broadcast. IEEE NaN/infinity behavior is preserved, including floating-point division by zero.

### Reductions, algebra, and comparison

| Function | Signature | Behavior |
| --- | --- | --- |
| `sum` | `NDArray -> float` | Whole-array sum; empty zero |
| `mean` | `NDArray -> float` | Whole-array mean; empty error |
| `min` | `NDArray -> float` | Whole-array minimum; empty error |
| `max` | `NDArray -> float` | Whole-array maximum; empty error |
| `dot` | `NDArray -> NDArray -> float` | Equal-length rank-one vectors; empty zero |
| `matvec` | `NDArray -> NDArray -> NDArray` | `[m;n]` × `[n]` → `[m]` |
| `matmul` | `NDArray -> NDArray -> NDArray` | `[m;k]` × `[k;n]` → `[m;n]` |
| `equals` | `NDArray -> NDArray -> bool` | Same shape and exact values; NaNs unequal, signed zeros equal |
| `allClose` | `NDArray -> NDArray -> float -> float -> bool` | Array, other, absolute tolerance, relative tolerance |

Reductions propagate NaN; different traversal/kernel paths can round differently. Zero inner dimensions produce zero-filled matrix/vector results. Tolerances must be finite and nonnegative. `allClose` uses a symmetric `abs(a-b) <= atol + rtol*max(abs(a),abs(b))` comparison, treating matching infinities as equal and NaNs as unequal. Different shapes compare false; equal empty shapes compare true.

Ordinary FScript equality compares `Id` as part of the visible snapshot, so separate equal arrays remain distinct:

```fsharp
let a = NDArray.ofList [1.0;2.0]
let b = NDArray.copy a
(a = a, a = b, NDArray.equals a b, NDArray.allClose a b 0.0 0.0)
// (true, false, true, true)
```

See the executable [views example](../../examples/views.fss) and [storage diagram](../architecture/overview.md) for explicit mutation and aliasing. Formatting exposes metadata only; use `toList` or `toRows` when you need numerical contents.

## Embedding

Reference `FScript.Numerics.Extensions`. Use these public helpers from namespace `FScript.Numerics.Extensions`:

| Helper | Purpose |
| --- | --- |
| `NumericsExtension.CreateRegistration()` | Complete SDK type/function registration |
| `NumericsExtension.Wrap(NDArray)` | SDK value with stable Id, metadata snapshot, and actual array payload |
| `NumericsExtension.Unwrap(ScriptValue)` | Validate descriptor, snapshot, payload type; return the same NDArray instance |
| `NumericsProvider.Provide()` | Attributed provider discovery entry point |

An embedder that separately references Runtime adapts the SDK registration using `ExtensionBridge.toLanguageRegistration` before passing it to registration-aware hosting APIs. The production extension itself has no Language or Runtime reference.

The bridge validates incoming/outgoing values against SDK schemes, checks the host runtime before/after invocation, and translates argument, shape, arithmetic-validation, and invalid-payload failures into descriptive `ExtensionException` instances retaining their inner cause. Host runtime limits and interpreter error reporting remain owned by the embedding host.
