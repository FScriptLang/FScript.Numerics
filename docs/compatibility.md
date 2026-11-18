# Versions, loading, and compatibility

All six packages target .NET 11 and share one repository version. Companion project dependencies are packed as exact `[version]` constraints. `FScript.Numerics.Extensions` depends on published MagnusOpera.FScript.Extensibility 2.0.0; numerical libraries have no SDK, FSharp.Core, Language or Runtime dependency. Test tooling alone references Runtime and the published CLI/server 0.94.3.

SDK 2 has an unsigned assembly identity of `2.0.0.0`. Providers built for SDK 1 must be rebuilt; this provider no longer targets the SDK 1 ABI. All eight numerical descriptors are monomorphic and pass an explicit empty `TypeParameters` list. Generic SDK 2 descriptors cannot impersonate these numerical values.

Use the aggregate extension once in an environment. It registers NDArray and seven opaque fitted/factorization types, with monomorphic schemes for all functions. Numerical BLAS/LAPACK/Statistics/ML packages contain no competing providers. They consume one concrete NDArray implementation, not interfaces or duplicated wrappers.

FScript's NuGet environment resolves all configured packages as one dependency graph and loads their numerical assemblies once. SDK/FSharp.Core contracts are shared with the host. Duplicate incompatible assemblies or registration names cause environment initialization to fail; they do not silently create incompatible NDArray types. Each manifest-local cache records its restored graph. Restart the host/editor environment after changing packages.

An embedding app that already references numerical CLR types should use typed registrations in its own assembly context:

```csharp
var sdk = NumericsExtension.CreateRegistration();
var registration = ExtensionBridge.toLanguageRegistrationWithPermissions(
    FScript.Language.HostPermissions.Native, sdk);
var loaded = ScriptHost.loadSourceWithRegistration(registration,
    "NDArray.sum (NDArray.ofList [1.0;2.0])");
```

The embedding host explicitly grants `HostPermissions.Native` to call SDK provider functions, including managed numerical operations. The default bridge grants no permissions and leaves these functions unavailable; the provider cannot grant itself access.

Only the embedding application references FScript.Runtime. Providers reference the SDK and numerical libraries. Use `NumericsExtension.Wrap/Unwrap` for arrays and `ModelInterop.Wrap/Unwrap<T>` for fitted models. Payload type, descriptor, and snapshot metadata are checked. Do not cast private NuGet-context payloads to a separately loaded default-context NDArray type; use one host-owned registration or transport through SDK values.

`make package-smoke-tests` packs the actual packages, restores them from an isolated local feed through published FScript, executes every example, then checks the published language server and debugger with that same manifest. It uses separate restore caches so rebuilding version 0.1.0 cannot accidentally reuse earlier development bits. No interpreter modifications are required.

The debugger reveals bounded visible metadata, not storage or fitted coefficients. Watches may read fields but cannot run arbitrary host functions. Ordinary equality is stable per-instance identity, including through functions, containers, closures, tasks, and repeated host wrapping. Explicit numerical equality is separate.

This repository does not publish automatically during development. Release tooling creates six package/symbol pairs, changelog-derived notes and a draft; publication requires explicit release approval. Compatible public API changes follow semantic versioning; this initial 0.x expansion changes dtype behavior and broadcasting while retaining explicit mutation.
