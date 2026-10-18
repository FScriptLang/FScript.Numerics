# Contributor requirements

Keep all configuration in this repository. The numerical core must not reference FScript or FSharp.Core; production extensions must not reference Language or Runtime.
Run `make build test smoke-tests pack verify-changelog`. Builds must be warning-free. Features and fixes require meaningful automated coverage and corresponding documentation.
Every ordinary commit must update CHANGELOG.md with a concise single-line Unreleased bullet. Release commits may empty Unreleased.
All projects share one version, initially 0.1.0. Prepare releases with `make release-prepare version=X.Y.Z` (preview: `dryrun=true`), push commit/tag together, wait for CI's draft, verify changelog notes and artifacts, then publish that draft. NuGet publication uses this repository's NUGET_KEY. Do not publish automatically during implementation.
