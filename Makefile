.PHONY: build test smoke-tests pack verify-changelog release-prepare reference verify-reference benchmark package-smoke-tests
build:
	dotnet build FScript.Numerics.slnx -c Release

test: verify-reference
	python3 .github/scripts/verify_native.py
	dotnet test FScript.Numerics.slnx -c Release
	python3 .github/scripts/test_release.py

smoke-tests: build
	python3 .github/scripts/native_failure.py
	python3 .github/scripts/smoke.py
	python3 .github/scripts/lsp_smoke.py

pack:
	@for project in src/*/*.csproj; do dotnet pack "$$project" -c Release -o .out/packages $(if $(version),-p:Version=$(version)) || exit 1; done
	python3 .github/scripts/verify_packages.py $(version)

reference:
	dotnet run --project tools/Reference -c Release

verify-reference:
	dotnet run --project tools/Reference -c Release -- --check

benchmark:
	dotnet run --project benchmarks/FScript.Numerics.Benchmarks -c Release

package-smoke-tests: pack
	$(if $(version),NUMERICS_VERSION=$(version)) python3 .github/scripts/package_smoke.py

verify-changelog:
	python3 .github/scripts/changelog.py

release-prepare:
	python3 .github/scripts/release.py "$(version)" "$(dryrun)"
