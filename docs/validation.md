# Expansion validation

M0–M5 are implemented with explicit Set/set mutation retained. The original validation used .NET SDK 10.0.401, SDK contracts 1.0.0, and published FScript CLI/Runtime/VS Code server 0.83.1. The existing interpreter and SDK repositories are unchanged.

- Warning-free Release solution build.
- 114 .NET numerical/registration/inference tests and five release-tool tests.
- All nine checked-output examples through the published CLI and adjacent manifest.
- Six NuGet packages plus six symbol packages; exact companion dependency graphs and no production Runtime/Language dependency.
- Pinned native package hashes and actual machine headers, float32/64 native reconstruction, C/F/offset/strided input adaptation, singularity and concurrency tests.
- Fresh-process missing-native diagnostics with registration still available.
- Every example through restored local NuGet artifacts, with isolated caches to prevent stale development packages.
- Published LSP completion, visible-field/model hover, authoritative diagnostics and privacy checks against that package manifest.
- Published CLI DAP launch, actual native execution, breakpoint, million-element array metadata inspection, model privacy, and normal termination.
- Release-preparation initial/subsequent dry runs and actual preparation in temporary Git fixtures; no commit/tag/publication in the working repository.
- Changelog gate, checked generated registration reference, source whitespace checks, Python and Node syntax checks.
- Recorded copy/allocation/thread diagnostics in [benchmark results](benchmark-results.md).

The numerical/native and package/editor/debugger validations pass on macOS arm64 and in a Linux x64 .NET SDK container with libgfortran5 installed. Linux container validation uses a disposable source snapshot and temporary Git metadata; it never writes into the repository. The final upper-rate Poisson convergence regression is additionally covered by the macOS numerical suite.

Windows x64 CI is configured with MinGW64 GNU runtime prerequisites but was not executed locally. macOS x64 is missing from the selected vendor package; linux-arm64 and win-x86 binaries are mislabeled x64 and deliberately rejected. These are native-backend limits; managed NDArray/statistics/scaler/clustering operations do not load that backend. See [native platform requirements](native-backends.md).

No packages or product releases were published by this change.

## SDK 2 and .NET 11 validation

The migration is validated on macOS arm64 with .NET SDK 11.0.100, Extensibility
2.0.0, System.Numerics.Tensors 11.0.0, and published FScript Runtime/CLI/server
0.94.3. The historical platform and timing results above describe the earlier
.NET 10 baseline.

- Warning-free Release solution build; 124 .NET tests and five release-tool tests pass.
- SDK 2 regression tests cover all eight monomorphic descriptors, generic-descriptor impersonation, published-host round trips, and explicit Native permission grants.
- All nine checked-output examples pass with local DLL discovery and again through isolated local NuGet package restoration.
- Native hashes and machine headers, fresh-process missing-native diagnosis, and the checked 137-function registration reference pass.
- All six `net11.0` packages and symbol packages pass exact dependency checks; production libraries retain their dependency boundaries.
- Published LSP completion/hover/diagnostics and CLI debugger execution/opaque inspection pass against the restored package environment.
- The optional compiler comparison builds with .NET 11 and verifies four workloads over 32 seeds, mutation, aliasing, retained failures, and closed-method JIT validity against the sibling FScript checkout.
- Workflow lint, changelog preflight, and whitespace checks pass. No release tags or package publication are part of this validation.
