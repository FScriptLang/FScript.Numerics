# FScript extension and embedding

The production `FScript.Numerics.Extensions` assembly references Extensibility 1.0.0 and the five numerical libraries. It never references Language or Runtime. `[FScriptExternProvider]` marks the zero-argument `NumericsProvider.Provide` method returning `ExtensionRegistration`.

`NumericsExtension.CreateRegistration()` supplies descriptors, no union declarations, and authoritative monomorphic `ExternalFunction` schemes. `NDArray` is nominal and opaque; script literals cannot manufacture it. The aggregate registers each type once, even though every module exchanges NDArray values. See [complete function/type tables](registration.md) and [operation policies](operations.md).

| Structure / helper | Fields or parameters | Purpose |
| --- | --- | --- |
| `ExtensionRegistration` | Types, Unions, Functions | One environment's type definitions and callable registrations |
| `HostTypeDescriptor` | Name, Fields, HostManaged=true | Script-visible immutable snapshots and nominal opaque value identity |
| `ExternalFunction` | Name, Scheme, Impl, Arity | Curried monomorphic API and host invocation handler |
| `FunctionScheme` | Forall([], ScriptType) | Authoritative argument and result types; dynamic dtype checks remain in the handler |
| `NumericsExtension.Wrap` | NDArray | SDK value holding snapshot plus the actual core array payload |
| `NumericsExtension.Unwrap` | ScriptValue | Validates descriptor, snapshot and concrete payload before exposing a core array to hosts |
| `ModelInterop.Wrap` | Supported fitted object | SDK opaque fitted value with stable weak-key Id |
| `ModelInterop.Unwrap<T>` | ScriptValue | Validates model identity/descriptor/payload/metadata; rejects another model type |

`Sdk.Declare` is internal C# adaptation for FSharp lists/maps/delegates; it builds the SDK records, validates every argument/result, and calls `CheckRuntime` around execution. Argument, shape, overflow and numerical validation errors become `ExtensionException` with an operation-qualified message and retained inner cause. Interpreter type errors occur before execution where the registered scheme can reject a program.

SDK values transport the concrete hidden array/model unchanged. Visible fields contain only layout and bounded fitted metadata; storage, RNG internals, coefficients, native pointers/workspaces and reflection handles are never exposed. All identity tables use weak keys. Wrapping the same object again gives the same Id; a view/copy/new fit is a distinct object and Id. Array storage mutation never changes its layout snapshot.

Host-managed fields remain readable/destructurable; construction and both record updates are rejected. Equality compares visible snapshots and therefore identity, not mutable numerical contents. `NDArray.equals` compares contents, `NDArray.equal` produces a broadcast mask, and `NDArray.allClose` applies explicit tolerances.

For embedding/load-context guidance and a registration example, see [compatibility](../compatibility.md). The [adjacent sample manifest](../../examples/fscript.toml) works for CLI, editor completion and debugger launches. Installing published packages uses the existing `[[runtime.packages]]` declaration; there is no separate numerical install command.

## Compiler purity and task-sharing contracts

The provider exposes optional BCL-only companion methods understood by FScript 0.84+ (`Pure`) and 0.85+ (thread safety). Extensibility 1.0.0 registration records are unchanged; older hosts ignore these methods.

- `FScriptPureFunctions()` is empty. Arrays can be mutated through shared views; observations depend on storage state. Factories return observable fresh identities. Shape, dtype, index, convergence, and native-loading failures must remain observable, even if their results are unused. Successful numerical arithmetic alone does not establish FScript's strong deterministic, effect-free, successfully terminating `Pure` promise.
- `FScriptPureTypes()` names the seven fitted/factorization types. Their owned fitted storage cannot be mutated through their public APIs; coefficient/factor access returns copies. Visible snapshots, equality, and formatting are stable after wrapping. This does not make fitting or prediction pure. `NDArray` is deliberately excluded, including read-only broadcast views, which may alias writable storage.
- `FScriptThreadSafeTypes()` names those same seven immutable fitted objects. It excludes `NDArray`; concurrent mutation through any storage alias requires application coordination.
- `FScriptThreadSafeFunctions()` identifies functions whose arguments contain only immutable script scalars/containers or immutable fitted objects. This includes array factories and scalar/seeded distributions, which use local state and return fresh arrays. Any NDArray argument, including one inside a list, excludes the operation: another task may write its storage. Thread-safe does not imply pure, total, or deterministic identity.

The function list is derived from authoritative argument signatures. New argument kinds default to unsafe until reviewed. No blanket claim is made for BLAS/LAPACK, mutable array readers, views, copying, or fitting on shared arrays. Disjoint arrays can be used independently; analysis remains conservative about potentially shared storage. These contracts add no per-call runtime checks or locking and do not change mutation, identity, validation, or numerical semantics.

## Typed array bindings and explicit Inline

The provider exposes typed CLR companions for 19 array operations: `arange`, `ofList`, `zeros`, `ones`, `full`, `copy`, `contiguous`, `transpose`, `addScalar`, `multiplyScalar`, `sum`, `dot`, `toList`, and float/int/bool `get`/`set`. These entry points use the existing SDK `HostManagedValue` representation. There is no `FScriptCompiledTypes` override, so unconverted operations continue to work through SDK dispatch on the very same values. Interpreters continue using the original handlers.

Typed entry points avoid general function-list lookup, ScriptValue argument lists, and repeated scheme/type-definition validation. They retain nominal descriptor/payload/snapshot checks, checked integer index/shape conversion, result copying, shared views, stable instance identity, dtype/shape/index failures, and operation-qualified ExtensionException causes. Static CLR signatures supply the primitive/list shape guarantees. Calls remain non-pure and unsafe for uncoordinated shared-array mutation.

`FScriptInlineFunctions()` selects only the six small getter/setter adapters. FScript's .NET backend can replace these exact forwarders with direct calls to the checked operations at IL level, without spill locals or an extra adapter frame. The public checked helper retains exception qualification; its EH implementation is not copied. The six checked getter/setter methods use direct try/catch bodies instead of allocating delegate closures. Larger array operations use typed calls without requesting expansion. This deliberately separates transport optimization from kernel optimization; it does not promise inlining of the entire NDArray core getter or removal of bounds/snapshot checks.

The companion contract is additive BCL metadata, leaving the Extensibility 1.0.0 API and 137 interpreter schemes unchanged. It requires an FScript compiler supporting explicit Inline metadata; older compilers ignore unknown companion methods. It does not by itself establish Native AOT support for all Numerics/native dependencies. Use the [comparison](../../benchmarks/FScript.Numerics.CompilerComparison/README.md) to measure SDK, typed, and explicitly expanded calls separately.

Array unwrapping validates the existing metadata fields in place instead of allocating a new snapshot map for comparison. All nine fields, exact key count, nested dimension lists, descriptor, and payload remain checked on every call. Dtype names use fixed enum literals. This is allocation removal, not memoization or a change to array mutability/purity; views and content mutations remain visible. Tests corrupt every descriptor field and check missing/extra keys and nested dimensions.

Typed shape/index inputs use IReadOnlyList<long>, avoiding an unnecessary compiler-side array conversion; the provider only reads them. Index narrowing still uses checked conversion into the core int array. Hot getters/setters use direct checked bodies, manual index conversion, and direct HostManagedValue unwrapping, avoiding delegate, LINQ-iterator, and ScriptValue transport allocations. A shared descriptor can be accepted by reference identity; structurally equivalent descriptors remain accepted through the existing equality fallback. Normal valid unwrapping has zero measured allocation, and getter regression tests bound allocation to its core index array. No payload validation cache or function memoization is added.
