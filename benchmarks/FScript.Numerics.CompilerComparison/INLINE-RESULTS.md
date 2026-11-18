# Typed array FFI and explicit IL expansion

Measured 2026-10-25 on Apple M3 / Arm64, macOS 27.0.1, SDK 10.0.401 and .NET 10.0.12. FScript compiler: `fad2476`, with opt-in IL expansion, 16/100-byte candidate screens, post-cleanup growth limits, and closure-site coverage. Numerics baseline: `553e771`, followed by the typed companions described here. No numerical kernel algorithm was changed.

Every process checks four workloads over 32 seeds for all five implementations, typed-only host representation, mutation, alias/copy behavior, mixed SDK/typed calls, metadata fields and identity, retained qualified failures, and CLR JIT validity. It checks measured checksums and verifies that opted-in adapters are absent from the expanded IL. Each process measures 100 calls per batch and 7 batches per implementation after warmup, without concurrent builds/tests. Tables report medians of three independent process medians and their ranges.

## Typed companions before snapshot-check optimization

These runs use the original snapshot reconstruction in Unwrap. They isolate SDK transport, typed calls without Inline, typed calls with Inline, and unchanged direct F# baselines. Raw artifacts: `.out/compiler-inline-{default,nontiered}{1,2,3}/`.

### Default JIT policy

| Workload | Implementation | ms / call | Process-median range (ms) | Bytes / call |
| --- | --- | ---: | ---: | ---: |
| `bulk32` | fscript-sdk | 0.036899 | 0.036195–0.037342 | 72,181 |
| `bulk32` | fscript-typed | 0.022010 | 0.021292–0.022192 | 48,565 |
| `bulk32` | fscript-inline | 0.022036 | 0.021338–0.022200 | 48,565 |
| `bulk32` | fsharp-ndarray | 0.001545 | 0.001510–0.001585 | 3,576 |
| `bulk32` | fsharp-array | 0.000274 | 0.000265–0.000275 | 1,192 |
| `bulk4096` | fscript-sdk | 0.122840 | 0.122240–0.123121 | 201,583 |
| `bulk4096` | fscript-typed | 0.116386 | 0.116020–0.116650 | 178,199 |
| `bulk4096` | fscript-inline | 0.115871 | 0.115600–0.116505 | 178,199 |
| `bulk4096` | fsharp-ndarray | 0.102618 | 0.102199–0.102895 | 133,624 |
| `bulk4096` | fsharp-array | 0.032667 | 0.032125–0.032977 | 131,408 |
| `bulk65536` | fscript-sdk | 1.654777 | 1.648901–1.658383 | 2,168,110 |
| `bulk65536` | fscript-typed | 1.631597 | 1.627179–1.632552 | 2,144,858 |
| `bulk65536` | fscript-inline | 1.632133 | 1.626417–1.635134 | 2,144,725 |
| `bulk65536` | fsharp-ndarray | 1.609080 | 1.608506–1.612219 | 2,100,043 |
| `bulk65536` | fsharp-array | 0.562153 | 0.559623–0.576383 | 2,097,829 |
| `indexed1024` | fscript-sdk | 2.758295 | 2.753671–2.766080 | 11,358,011 |
| `indexed1024` | fscript-typed | 1.481462 | 1.471611–1.493875 | 6,544,624 |
| `indexed1024` | fscript-inline | 1.484267 | 1.475375–1.494698 | 6,544,624 |
| `indexed1024` | fsharp-ndarray | 0.006724 | 0.006712–0.006728 | 49,448 |
| `indexed1024` | fsharp-array | 0.005219 | 0.005153–0.005235 | 16,624 |

### Tiered compilation / ReadyToRun disabled

| Workload | Implementation | ms / call | Process-median range (ms) | Bytes / call |
| --- | --- | ---: | ---: | ---: |
| `bulk32` | fscript-sdk | 0.024035 | 0.023730–0.024198 | 72,109 |
| `bulk32` | fscript-typed | 0.015302 | 0.015085–0.015375 | 48,699 |
| `bulk32` | fscript-inline | 0.015303 | 0.015123–0.015341 | 48,493 |
| `bulk32` | fsharp-ndarray | 0.001906 | 0.001864–0.001924 | 3,576 |
| `bulk32` | fsharp-array | 0.000387 | 0.000380–0.000389 | 1,232 |
| `bulk4096` | fscript-sdk | 0.164379 | 0.162641–0.164769 | 202,167 |
| `bulk4096` | fscript-typed | 0.156369 | 0.154327–0.156485 | 178,551 |
| `bulk4096` | fscript-inline | 0.155617 | 0.154027–0.156219 | 178,551 |
| `bulk4096` | fsharp-ndarray | 0.140354 | 0.138559–0.140422 | 133,624 |
| `bulk4096` | fsharp-array | 0.043832 | 0.043733–0.043946 | 131,448 |
| `bulk65536` | fscript-sdk | 2.249201 | 2.213610–2.251637 | 2,168,699 |
| `bulk65536` | fscript-typed | 2.228647 | 2.200077–2.233287 | 2,145,297 |
| `bulk65536` | fscript-inline | 2.228555 | 2.198652–2.232016 | 2,145,297 |
| `bulk65536` | fsharp-ndarray | 2.198165 | 2.169594–2.209481 | 2,100,206 |
| `bulk65536` | fsharp-array | 0.673937 | 0.669580–0.727776 | 2,097,880 |
| `indexed1024` | fscript-sdk | 3.442837 | 3.394654–3.443900 | 11,358,163 |
| `indexed1024` | fscript-typed | 1.832267 | 1.817642–1.850116 | 6,577,456 |
| `indexed1024` | fscript-inline | 1.831884 | 1.821787–1.835477 | 6,577,456 |
| `indexed1024` | fsharp-ndarray | 0.006655 | 0.006402–0.006668 | 49,448 |
| `indexed1024` | fsharp-array | 0.008120 | 0.008118–0.008135 | 16,664 |

Typed calls reduce default-JIT getter-loop time from 2.758 ms to 1.481 ms and allocations from 11.36 MB to 6.54 MB. Explicit IL expansion has no material additional steady-state benefit over typed calls in these runs: the JIT can already inline the tiny adapters. The improvement against SDK dispatch must therefore not be attributed to static IL expansion. Large bulk kernels are close to direct NDArray execution, while small operations still pay significant metadata-validation costs.

The additional static benefit is controlled generated IL and dependency reachability. Getter/setter adapter calls are replaced by checked-operation calls, without an extra adapter frame or spill locals. Numeric test kernels also demonstrate copied arithmetic/branches, constant cleanup, retained effects/failures, and bounded growth. Size policy is a conservative starting heuristic based on .NET 10 screening constants, not a claim of universally optimal native code.

## Next measured change: snapshot checks

The first comparison exposed repeated construction of the nine-field metadata map in Unwrap as a remaining allocation source. The next step compares existing fields directly on every call, including shape, strides, identity, dtype, and flags. This does not cache validation or mutable array contents. The final results below include both in-place checks and removal of delegate/LINQ/transport allocations from typed getters/setters.

## Final provider: in-place validation and lean checked getters

All nine metadata fields and exact field counts remain checked on every call. Shared immutable descriptors use a reference-equality fast path; structurally equivalent descriptors still work. Shape/stride lists are checked directly, dtype names are constant literals, and no snapshot map is reconstructed. Typed getters/setters use direct try/catch bodies, a manual checked index conversion, and direct host-value unwrapping rather than creating delegate closures or temporary SDK values. No validation/content cache or memoization is introduced.

Allocation regression tests verify zero allocation for repeated valid SDK unwrapping and no more than the core index-array allocation for a typed getter with pre-existing inputs. Forged fields, missing/extra keys, malformed dimension lists, descriptor copies, dtype errors, checked index overflow, mutation and aliasing still pass their correctness tests. These changes apply to both compiled and interpreted Numerics entry points; no FScript language/runtime interpreter code was changed.

Raw artifacts: `.out/compiler-fields-{default,nontiered}{1,2,3}/`. This comparison uses the same five-variant harness as above. Its SDK column also benefits from the new validation, so it is not the old provider baseline.

### Default JIT policy

| Workload | Implementation | ms / call | Process-median range (ms) | Bytes / call |
| --- | --- | ---: | ---: | ---: |
| `bulk32` | fscript-sdk | 0.039907 | 0.026411–0.040326 | 47,344 |
| `bulk32` | fscript-typed | 0.025298 | 0.012674–0.025908 | 23,600 |
| `bulk32` | fscript-inline | 0.025417 | 0.013030–0.025799 | 23,600 |
| `bulk32` | fsharp-ndarray | 0.012633 | 0.001559–0.013116 | 3,576 |
| `bulk32` | fsharp-array | 0.000257 | 0.000236–0.000263 | 1,192 |
| `bulk4096` | fscript-sdk | 0.117189 | 0.116673–0.118380 | 176,736 |
| `bulk4096` | fscript-typed | 0.112511 | 0.111844–0.114528 | 153,320 |
| `bulk4096` | fscript-inline | 0.114337 | 0.111633–0.117252 | 153,320 |
| `bulk4096` | fsharp-ndarray | 0.102786 | 0.102624–0.106647 | 133,624 |
| `bulk4096` | fsharp-array | 0.033597 | 0.032550–0.035046 | 131,408 |
| `bulk65536` | fscript-sdk | 1.642827 | 1.640857–1.644906 | 2,143,176 |
| `bulk65536` | fscript-typed | 1.669042 | 1.663018–1.711484 | 2,119,683 |
| `bulk65536` | fscript-inline | 1.669024 | 1.664062–1.709839 | 2,119,683 |
| `bulk65536` | fsharp-ndarray | 1.613237 | 1.611832–1.615940 | 2,099,987 |
| `bulk65536` | fsharp-array | 0.503390 | 0.495274–0.508141 | 2,097,741 |
| `indexed1024` | fscript-sdk | 1.367900 | 1.345485–1.376540 | 5,049,432 |
| `indexed1024` | fscript-typed | 0.101001 | 0.100049–0.102933 | 88,872 |
| `indexed1024` | fscript-inline | 0.099777 | 0.099577–0.100068 | 88,872 |
| `indexed1024` | fsharp-ndarray | 0.006724 | 0.006713–0.006729 | 49,448 |
| `indexed1024` | fsharp-array | 0.005184 | 0.005142–0.005201 | 16,624 |

### Tiered compilation / ReadyToRun disabled

| Workload | Implementation | ms / call | Process-median range (ms) | Bytes / call |
| --- | --- | ---: | ---: | ---: |
| `bulk32` | fscript-sdk | 0.017245 | 0.017045–0.017389 | 47,272 |
| `bulk32` | fscript-typed | 0.008834 | 0.008739–0.008890 | 23,528 |
| `bulk32` | fscript-inline | 0.008875 | 0.008731–0.008897 | 23,528 |
| `bulk32` | fsharp-ndarray | 0.001900 | 0.001869–0.001912 | 3,576 |
| `bulk32` | fsharp-array | 0.000380 | 0.000375–0.000383 | 1,232 |
| `bulk4096` | fscript-sdk | 0.158072 | 0.157947–0.158568 | 177,320 |
| `bulk4096` | fscript-typed | 0.146457 | 0.146100–0.146638 | 153,576 |
| `bulk4096` | fscript-inline | 0.146186 | 0.145803–0.146355 | 153,576 |
| `bulk4096` | fsharp-ndarray | 0.138664 | 0.138325–0.138752 | 133,624 |
| `bulk4096` | fsharp-array | 0.043735 | 0.043721–0.043876 | 131,448 |
| `bulk65536` | fscript-sdk | 2.214329 | 2.212292–2.227957 | 2,143,906 |
| `bulk65536` | fscript-typed | 2.198677 | 2.188946–2.205395 | 2,120,083 |
| `bulk65536` | fscript-inline | 2.199620 | 2.189477–2.205342 | 2,120,083 |
| `bulk65536` | fsharp-ndarray | 2.184446 | 2.175905–2.193130 | 2,100,299 |
| `bulk65536` | fsharp-array | 0.668943 | 0.668620–0.669273 | 2,097,844 |
| `indexed1024` | fscript-sdk | 1.749955 | 1.710744–1.756575 | 5,049,584 |
| `indexed1024` | fscript-typed | 0.181039 | 0.180620–0.181943 | 88,936 |
| `indexed1024` | fscript-inline | 0.181406 | 0.179732–0.181630 | 88,936 |
| `indexed1024` | fsharp-ndarray | 0.006437 | 0.006416–0.006712 | 49,448 |
| `indexed1024` | fsharp-array | 0.008130 | 0.008128–0.008146 | 16,664 |

The default-JIT getter loop drops from the earlier SDK provider's 2.758 ms / 11.36 MB to about 0.100 ms / 88.9 KB with typed expanded calls: roughly 27.6× faster and 128× less allocation in this microbenchmark. Against the updated SDK column, typed calls are about 13.7× faster. The direct F# NDArray baseline is still around 0.0067 ms; per-access host metadata checks remain a real cost.

There is still no clear general steady-state gain from the tiny adapter expansion alone over typed-without-Inline. The major improvement is removing transport and validation allocations, not copying a large body into the caller. This supports keeping expansion bounded and checked kernels out of the caller.

Large bulk timings show little benefit and some run-to-run ordering differences; they do not establish a universal improvement. Default-JIT tiny NDArray baselines shifted materially between the two experiments, despite unchanged numerical core algorithms. Do not attribute cross-experiment timing changes solely to one optimization or compare tiny rows without considering JIT policy. Simultaneous variant comparisons and the allocation regression checks provide stronger evidence here.

These are warmed synchronous microbenchmarks on one development machine, without confidence intervals. They do not measure Native AOT, native BLAS/LAPACK, tasks, startup, or whole-application performance. Preparation/compilation, reflection, validation, initialization, and warmup are excluded from measured execution. All measured checksums match, including all six final processes.

## Reading the F# comparison

The original FScript/F# workload comparison shows F# allocating 24 bytes for the captured numeric step versus FScript's 192 bytes, even though both tail-recursive kernels become loops. A useful compiler follow-up is liveness analysis after tail-loop lowering: remove dead recursive captures and unused curried scaffolding while preserving evaluation of effectful arguments. This must work from escapes/call graphs, not a blanket assumption that every function can be flattened.

F# arrays also avoid FScript's extra nominal/structural record adaptation in the record pipeline. A targeted optimization would forward field reads through non-escaping record conversions, then eliminate dead copies when all observations are proven safe. External getters and constructors require their existing contracts; mutable or failing FFI must not be silently elided. Neither follow-up is implemented or claimed as a speedup in this report. Existing FScript integer CSE already emits fewer multiplication sites than F# in the numeric loop, so instruction counts alone are not a ranking.

Programs without Inline metadata are unaffected by the new IL pass: the original FScript comparison DLL remains byte-identical to its pre-feature output (`33aa0d82715850866d167d836873280ea6d26cdaa46a8c00aabbcfa868a4cffd`).
