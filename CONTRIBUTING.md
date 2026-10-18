# Contributing

This is an independent repository inside the FScript umbrella. Keep its solution, packages, tests, CI, and release metadata local to this directory. Use the SDK pinned in `global.json` and published package versions from `Directory.Packages.props`.

Run `make build test smoke-tests pack package-smoke-tests verify-changelog` before submitting changes. Builds treat warnings as errors. Add tests for each new feature and regression; update API/architecture documentation when behavior changes. Every commit to main updates `CHANGELOG.md` with a concise, single-line Unreleased bullet. Release commits may reset Unreleased to empty.

Core tests cover layouts, copying, indexing, shared storage, strided/contiguous numerical paths, IEEE edge cases, and empty arrays. Extension tests use published Runtime 0.83.1 to cover descriptors, inference, value transport, mutation, errors, and dependency boundaries. Smoke tests run every example using the pinned CLI and compare checked-in output without generated identities. They also download the published 0.83.1 VSIX once into `.out/` and verify completion, hover, diagnostics, and manifest loading in its packaged language server. Set `FSCRIPT_RELEASE_SERVER_DIR` to an already extracted server directory to run that check offline. Release-tool tests use temporary Git repositories and never mutate the real repository.

## Release preparation

All six NuGet packages share `Directory.Build.props`'s version. The extension requires exactly the corresponding core version. The initial release is **0.1.0**. No signing or website deployment is involved.

1. Commit reviewed implementation, documentation, and Unreleased notes. Confirm a clean tree.
2. Preview with `make release-prepare version=0.1.0 dryrun=true`. This validates inputs without writing files, committing, tagging, or publishing.
3. Run `make release-prepare version=0.1.0`. It validates all gates, moves notes into the release section, updates the shared version, creates a preparation commit, and adds an annotated tag. For the initial release, the compare link starts at the bootstrap commit; later releases compare successive numeric tags.
4. Push the preparation commit and tag together: `git push origin main --follow-tags`.
5. Wait for tag CI to build/test, create all six `.nupkg` and six `.snupkg` artifacts, and create a GitHub **draft** release. Confirm its notes exactly match the version section in CHANGELOG, including the final compare link, and review the artifacts.
6. Publish that existing draft. The published-release workflow downloads the tagged artifacts and publishes all six packages and symbols to NuGet. Configure this repository's `NUGET_KEY` Actions secret with permission to publish all six package IDs before this step. Do not paste secret values into issues or logs.

Do not bypass the draft or manually build replacement release artifacts. PR/main CI validates builds, tests, smoke examples, package dependency graphs, and changelog discipline. This initial implementation does not automatically publish 0.1.0.
