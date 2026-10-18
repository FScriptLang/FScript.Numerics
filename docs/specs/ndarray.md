# .NET API

Install `FScript.Numerics` and use namespace `FScript.Numerics`. All numerical values are `double`, represented by `DType.Float64`. Both libraries require .NET 10.

## Shape and array metadata

`Shape` is sealed, immutable, implements `IReadOnlyList<int>` and `IEquatable<Shape>`, and compares dimensions structurally. `new Shape(params int[] dimensions)` copies its input. `ToArray()` returns another copy. Negative dimensions and excessive nonempty products are rejected. `new Shape()` represents a scalar; `new Shape(0)` is an empty vector.

| Member | Type | Purpose |
| --- | --- | --- |
| `Shape.Rank` / `Count` | `int` | Number of dimensions |
| `Shape.Size` | `int` | Product of dimensions; scalar one, empty zero |
| `Shape[index]` | `int` | Read a dimension |
| `Shape.ToArray()` | `int[]` | Independent dimension copy |
| `Shape.Equals`, `GetHashCode`, `ToString` | standard .NET methods | Structural equality, hashing, readable dimensions |

`NDArray` is sealed, with private storage. Create instances using static factories.

| Property | Type | Purpose |
| --- | --- | --- |
| `Shape` | `Shape` | Immutable dimensions |
| `Strides` | `IReadOnlyList<int>` | Immutable **element** strides |
| `Rank` | `int` | Number of dimensions |
| `Size` | `int` | Logical element count |
| `Offset` | `int` | Starting element offset into shared storage |
| `DType` | `DType` | Always `Float64` |
| `IsContiguous` | `bool` | Logical row-major layout, ignoring singleton strides |

## Factories

| Method | Result and validation |
| --- | --- |
| `Scalar(double value)` | Rank-zero single element |
| `FromArray(double[] values)` | Rank-one copy; rejects null |
| `FromArray(ReadOnlySpan<double> values, Shape shape)` | Shaped copy; count must equal shape size |
| `FromRows(params double[][] rows)` | Rank-two copy; rejects null or ragged rows |
| `Zeros(Shape shape)` | New row-major storage filled with zero |
| `Ones(Shape shape)` | New storage filled with one |
| `Full(Shape shape, double value)` | New storage filled with value |
| `Arange(double start, double stop, double step)` | Stop-exclusive vector; finite arguments and nonzero step required |
| `Identity(int size)` | Square identity matrix; nonnegative size |

`FromRows()` produces `[0,0]`; `FromRows([], [])` preserves shape `[2,0]`. NaNs and infinities are valid stored values. `Arange` alone requires finite numeric inputs; a direction that disagrees with the bounds produces an empty vector. Floating-point endpoints follow double rounding, with values at or beyond the stop omitted. Excessive requested counts throw before allocation.

## Access and views

| Method | Behavior |
| --- | --- |
| `Get(params int[] indices)` | Read one element; exactly one nonnegative in-bounds index per dimension |
| `Set(double value, params int[] indices)` | Explicitly mutate exactly one element |
| `Transpose()` | Zero-copy view reversing every axis; rank zero/one also supported |
| `Permute(params int[] axes)` | Zero-copy view; complete unique permutation required |
| `Slice(int axis, int start, int stop, int step = 1)` | Rank-preserving shared view; `0 <= start <= stop <= dimension`, step positive |
| `Reshape(Shape shape)` | Shared view with equal size; contiguous input required |
| `Copy()` | Independent contiguous array in logical row-major order |
| `AsContiguous()` | Same instance when contiguous; otherwise `Copy()` |
| `TryGetContiguousSpan(out ReadOnlySpan<double> span)` | True with read-only logical contents when contiguous; false and default span otherwise |
| `ToArray()` | Independent flattened row-major `double[]`, any rank |
| `ToRows()` | Independent `double[][]`; rank two required |

A scalar is indexed with no indices, for example `scalar.Get()` and `scalar.Set(2.0)`. Empty arrays have no valid element index. Slice steps may cause a checked stride overflow even on an empty result. `Reshape` never silently copies; explicitly call `Copy()` or `AsContiguous()` first. Read-only spans share storage, so callers must synchronize concurrent mutations.

## Arithmetic, reductions, and algebra

All returned NDArrays have new contiguous storage. Operands remain unchanged.

| Method | Requirement / result |
| --- | --- |
| `Add(NDArray other)` | Equal shapes; elementwise addition |
| `Subtract(NDArray other)` | Equal shapes; elementwise subtraction |
| `Multiply(NDArray other)` | Equal shapes; elementwise multiplication |
| `Divide(NDArray other)` | Equal shapes; elementwise division |
| `Add(double scalar)` | Right-hand scalar addition |
| `Subtract(double scalar)` | Right-hand scalar subtraction |
| `Multiply(double scalar)` | Right-hand scalar multiplication |
| `Divide(double scalar)` | Right-hand scalar division |
| `Negate()` | Elementwise unary minus |
| `Sum()` | Whole-array sum; empty returns zero |
| `Mean()` | Whole-array mean; empty throws |
| `Min()` | Whole-array minimum; empty throws |
| `Max()` | Whole-array maximum; empty throws |
| `Dot(NDArray other)` | Rank-one equal-length vectors; double scalar; empty returns zero |
| `Matvec(NDArray vector)` | `[m,n]` × `[n]` → `[m]` |
| `Matmul(NDArray other)` | `[m,k]` × `[k,n]` → `[m,n]` |
| `ContentEquals(NDArray other)` | Equal shape and element values; NaN unequal, signed zeros equal |
| `AllClose(NDArray other, double absoluteTolerance, double relativeTolerance)` | Equal shape and symmetric per-element tolerance comparison |

Array arithmetic requires exactly matching shapes, including rank. A rank-zero NDArray never broadcasts implicitly; use the scalar overload explicitly. Matrix/vector multiplication requires the documented ranks, including when one dimension is zero. A zero inner dimension produces zeros of the resulting shape.

Arithmetic preserves IEEE double behavior: division by zero yields infinities or NaN; reductions propagate NaN. Empty `Mean`, `Min`, and `Max` throw `InvalidOperationException`. Optimized reductions can round differently from strided reductions.

`AllClose` requires finite, nonnegative tolerances and tests:

```text
abs(a - b) <= absoluteTolerance + relativeTolerance * max(abs(a), abs(b))
```

The comparison is symmetric, including an overflow-safe scaled calculation for extreme finite values. Equal infinities compare equal; other infinities and all NaNs compare unequal. Both numerical comparisons return false for different shapes and true for equal empty shapes. .NET `Equals` retains normal reference identity; use `ContentEquals` for numerical equality.

## Errors

Null arguments throw `ArgumentNullException`; invalid shapes, ranks, permutations, or shape mismatches throw `ArgumentException`; invalid indices, bounds, steps, and tolerances throw `ArgumentOutOfRangeException`. Unsupported non-contiguous reshape or rank-two conversion throws `InvalidOperationException`. Checked layout/count overflow throws `OverflowException`. Available memory can still limit otherwise valid allocations.

The [FScript bridge](fscript-api.md) translates argument, shape, and validation failures into SDK `ExtensionException` with the operation name and original cause. It does not change interpreter semantics.
