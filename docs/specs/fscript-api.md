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
