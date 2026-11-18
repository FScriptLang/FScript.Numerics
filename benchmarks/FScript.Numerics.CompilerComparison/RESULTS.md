# Numerical array comparison (historical SDK baseline)

These measurements precede the typed/Inline companions added after `553e771`. See [the updated comparison](INLINE-RESULTS.md).

Measured 2026-10-25 on Apple M3 / Arm64, macOS 27.0.1, .NET SDK 10.0.401 / runtime 10.0.12. Numerics core baseline: `b938992`; FScript compiler/harness: `a2eefc6` (0.85.0 sources). Only provider contracts, tests, and this comparison were added; numerical kernels and the compiler were unchanged.

Four workloads matched direct F# NDArray and F# array results over 32 seeds in every process. Compiled mutation, alias/copy, unused-failure, and closed-method JIT checks passed. Every measured batch checksum matched. No generated program references FScript.Language or FScript.Runtime; SDK Extensibility/FSharp.Core transport is present and included in execution costs.

Each policy was measured in three independent processes, 100 calls per batch, 7 batches per implementation, after warmup. Tables show medians of process medians, process-median timing ranges, and median bytes allocated per call. Full per-batch results and IL are in `.out/compiler-comparison-{default,nontiered}{1,2,3}/`. See [methodology](README.md).

## Default JIT policy

| Workload | Implementation | ms / call | Process-median range (ms) | Allocated bytes / call |
| --- | --- | ---: | ---: | ---: |
| `bulk32` | fscript-sdk | 0.018494 | 0.017257–0.018776 | 71,628 |
| `bulk32` | fsharp-ndarray | 0.001481 | 0.001377–0.001487 | 3,576 |
| `bulk32` | fsharp-array | 0.000228 | 0.000224–0.000246 | 1,192 |
| `bulk4096` | fscript-sdk | 0.115901 | 0.115320–0.116768 | 201,686 |
| `bulk4096` | fsharp-ndarray | 0.096642 | 0.095990–0.096750 | 133,624 |
| `bulk4096` | fsharp-array | 0.028644 | 0.027393–0.028747 | 131,408 |
| `bulk65536` | fscript-sdk | 1.609502 | 1.608863–1.615281 | 2,168,233 |
| `bulk65536` | fsharp-ndarray | 1.578097 | 1.572583–1.580884 | 2,100,114 |
| `bulk65536` | fsharp-array | 0.705920 | 0.473378–0.714462 | 2,097,840 |
| `indexed1024` | fscript-sdk | 2.746116 | 2.723870–2.800321 | 11,358,011 |
| `indexed1024` | fsharp-ndarray | 0.006627 | 0.006407–0.006644 | 49,448 |
| `indexed1024` | fsharp-array | 0.005055 | 0.004884–0.005097 | 16,624 |

## Tiered compilation and ReadyToRun disabled

| Workload | Implementation | ms / call | Process-median range (ms) | Allocated bytes / call |
| --- | --- | ---: | ---: | ---: |
| `bulk32` | fscript-sdk | 0.023921 | 0.023570–0.024608 | 72,212 |
| `bulk32` | fsharp-ndarray | 0.001880 | 0.001725–0.001928 | 3,576 |
| `bulk32` | fsharp-array | 0.000376 | 0.000327–0.000376 | 1,232 |
| `bulk4096` | fscript-sdk | 0.157392 | 0.157335–0.163824 | 202,270 |
| `bulk4096` | fsharp-ndarray | 0.133431 | 0.133430–0.139951 | 133,624 |
| `bulk4096` | fsharp-array | 0.039181 | 0.039143–0.040298 | 131,448 |
| `bulk65536` | fscript-sdk | 2.197227 | 2.196815–2.316873 | 2,168,956 |
| `bulk65536` | fsharp-ndarray | 2.164168 | 2.160971–2.272650 | 2,100,408 |
| `bulk65536` | fsharp-array | 0.795185 | 0.793055–0.853680 | 2,097,874 |
| `indexed1024` | fscript-sdk | 3.435590 | 3.416620–3.479562 | 11,358,163 |
| `indexed1024` | fsharp-ndarray | 0.006662 | 0.006371–0.006667 | 49,448 |
| `indexed1024` | fsharp-array | 0.008126 | 0.008117–0.008147 | 16,664 |

Default-policy results show about 2% overhead against direct NDArray calls for the 65,536-element bulk pipeline, about 20% at 4,096 elements, and roughly 12.5× at 32 elements. The 1,024-call getter loop is roughly 414× slower than direct F# NDArray calls and allocates about 11.36 MB per invocation versus 49 KB. This is a fine-grained SDK-boundary problem; it does not show the numerical arithmetic itself running hundreds of times slower.

Plain F# arrays are a different representation/kernel baseline. They are faster in these workloads, but their large-array timing varies materially across processes (0.47–0.71 ms with default policy). The NDArray dot path uses TensorPrimitives on contiguous float64 storage; array creation and scalar arithmetic are also included here. No conclusion about native BLAS/LAPACK, strided multidimensional arrays, or ML follows from this comparison.

## Inlining decision

**Direct typed FFI is the useful next step; forcibly inlining the SDK bridge is not.** The resolved IL for each SDK call still performs provider lookup (`SdkInterop.Find`), constructs SDK value arrays/FSharp lists, encodes arguments, invokes `ExternalFunction.Impl` through delegates, and decodes the result. The extension additionally validates arguments/results, reconstructs type definitions, and checks host snapshots. A getter's useful work is tiny compared with that boundary.

FScript already supports exact CLR methods/types through `FScriptCompiledFunctions` and `FScriptCompiledTypes`. A suitable Numerics projection could replace that bridge with direct calls and give the JIT a chance to inline small typed adapters/getters. Mutation and possible failure do not prohibit inlining, so this does not require falsely declaring NDArray operations pure. Index/shape/dtype checks, aliasing, identity, exception behavior, and evaluation order would still need preservation.

That projection has not been implemented or benchmarked here; its speedup is an inference from the boundary costs, not a measured result. It must cover all reached NDArray operations consistently: the current compiler explicitly rejects mixed SDK/native host-type projections (`FSC6009`). Metadata fields and stable identity also need matching CLR representations; raw NDArray is not automatically a complete replacement for its SDK descriptor. Arbitrary cross-assembly IL-body copying would add complexity without first removing SDK transport.

For large bulk kernels, direct-call inlining alone has little time left to recover in this test. For fine-grained getters it could matter greatly once transport is removed. Reducing SDK lookup/validation allocations is another possible path when retaining legacy transport. Neither optimization is silently applied by this benchmark.

## Contracts and limits

No numerical function is declared Pure: mutable storage, observable fresh identities, and input/native failures prevent that blanket guarantee. Fitted-object types declare stable immutable observations; no factory or method inherits function purity. Thread-safe factories/scalar operations remain distinct from mutation safety for shared arrays. Existing mutation and identity semantics were retained.

These are development-machine synchronous microbenchmarks, not confidence intervals or whole-application guarantees. Compilation, reflection, validation, initialization, and warmup are excluded. Allocations use the calling thread; no task/native-BLAS throughput is measured. IL dumps describe code before further JIT optimization. The F# dump includes the runner assembly; workload methods are under `FScript.Numerics.CompilerComparison.Reference`.

FScript source SHA-256: `F5C817E07602CB9443BD997369C454E70F074F0154E0AB79580212143193EA02`.
