# Compiled numerical arrays: FScript and F#

This optional comparison compiles the real Numerics provider into an FScript .NET library and calls its exported native functions. It compares the same algorithm with F# calling NDArray directly, and with ordinary F# `double[]` arrays. Neither interpreter is timed. It uses the provider's real purity and typed/Inline declarations. No replacement intrinsic or false purity promise is introduced.

It requires the .NET 10 SDK and a local FScript checkout containing `benchmarks/FScript.DotNet.Comparison` (the reusable measurement and IL-inspection harness). By default that checkout is a sibling of this repository. It is deliberately outside the regular solution so numerical builds and packages do not depend on a compiler source checkout.

```sh
make verify-compiler-comparison
make benchmark-compiler
# Different checkout / batch size / output directory:
make benchmark-compiler fscript=/absolute/path/to/FScript iterations=100 rounds=7 output=.out/arrays
```

The Make target builds Release, then runs with `DOTNET_TieredCompilation=0 DOTNET_ReadyToRun=0`. For default runtime policy after building:

```sh
env -u DOTNET_TieredCompilation -u DOTNET_ReadyToRun dotnet benchmarks/FScript.Numerics.CompilerComparison/bin/Release/net11.0/FScript.Numerics.CompilerComparison.dll 100 7 .out/arrays-default
```

Repeat in separate processes on an idle machine. Output includes JSON batches, timing medians/ranges, per-thread allocations, checksums, source hash, JIT settings, generated managed dependencies, and resolved FScript/F# IL. Generated binaries and IL are separated into `fscript-sdk/`, `fscript-typed/`, and `fscript-inline/` subdirectories. Generated output is ignored by Git. The DLL copies are inspection artifacts, not a standalone published application or native deployment bundle.

## Workloads and validation

`bulk32`, `bulk4096`, and `bulk65536` create `[0, n)`, add a runtime scalar seed, multiply by 3, and take the vector's dot product with itself. The SDK variant makes four SDK calls; typed variants link the same operations through the production typed companions. F# NDArray uses exactly the same managed numerical methods. The plain-array baseline uses separate array maps and an indexed dot-product loop, preserving the intermediate collections. It is a representation/kernel baseline, not identical NDArray machinery. These cases do not use native BLAS.

`indexed1024` creates an array and sums its elements from a tail-recursive loop. FScript calls `NDArray.get` 1,024 times with a singleton index list; F# NDArray uses the core checked getter with singleton index arrays; ordinary F# arrays use native indexing. This intentionally exposes fine-grained boundary costs.

Each implementation is checked across 32 seeds, including negative values. Inputs keep the integral `double` checksums exactly representable. Timing adapts all five implementations through the same cached `Func<int64,int64>` convention, converting seeds to doubles and results back outside the exported function. This equal harness overhead remains included in each variant's timing. All measured batch checksums must match.

Additional compiled probes verify mutation, shared-view versus copy behavior, and retention of a failing unused index access. The harness JIT-prepares all closed generated methods and rejects a generated library that references FScript.Language or FScript.Runtime. It discovers the provider's actual `Pure` contracts and links the actual SDK handlers, including payload metadata checks. The SDK variant deliberately skips typed companions; the typed variant clears only Inline flags; the inline variant honors the complete production companions. Reported preparation time covers generating three measurement libraries and a separate typed-only validation program; it is not a single-compilation timing.

Compilation, reflection, export initialization, validation, and warmup are outside the measured batches. Each variant warms for at least 500 ms and 20 calls; variant order rotates/reverses, a full GC precedes each batch, and allocation is measured on the synchronous calling thread. Runtime allocations and GC during a batch remain included. There are no timing thresholds in validation. [The common harness](../../../FScript/benchmarks/FScript.DotNet.Comparison/README.md) explains the underlying protocol and limits; that link assumes the sibling checkout layout.

## What this says about inlining

The comparison includes three compiled FScript variants: original SDK dispatch, typed companions with expansion disabled, and typed companions with the actual per-binding Inline requests. F# NDArray and ordinary array baselines are unchanged. Correctness also checks mixed typed/SDK operations, metadata fields, and identity; the inline variant must contain no call to the opted-in adapters.

Typed array companions retain SDK HostManagedValue, so there is no mixed-native/SDK projection conflict. They avoid general provider lookup, argument-list transport, and repeated signature validation while preserving payload/snapshot checks and operation-qualified failures. No mutable array function becomes Pure.

The explicit IL pass can collapse the six forwarding getter/setter adapters into direct calls to the checked operations. Those helpers still perform descriptor, index, dtype, and failure handling. This is not complete inlining of the underlying NDArray getter, and it does not remove bounds checks. Comparing typed-without-Inline against typed-with-Inline isolates the extra transformation; a total gain against SDK must not be attributed entirely to IL expansion.

The public checked helpers contain exception handlers and call implementation-private helpers, so they are outside the first supported Inline subset. Further expansion would require correct EH/private-helper handling or a redesigned typed boundary. Production behavior must not be weakened merely to make an implementation eligible.

[RESULTS.md](RESULTS.md) records the historical SDK baseline; [INLINE-RESULTS.md](INLINE-RESULTS.md) records the updated comparison.

The final provider also validates snapshots in place, fast-paths shared immutable descriptors, and removes delegate/LINQ/transport allocations from typed getters/setters. All validation still occurs on every call; no mutable content or validation result is cached. The report separates the initial typed-binding measurements from this allocation-removal change.
