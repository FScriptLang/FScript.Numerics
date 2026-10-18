.PHONY: build test smoke-tests pack verify-changelog release-prepare
build:
	dotnet build FScript.Numerics.slnx -c Release

test:
	dotnet test FScript.Numerics.slnx -c Release
	python3 .github/scripts/test_release.py

smoke-tests: build
	python3 .github/scripts/smoke.py
	python3 .github/scripts/lsp_smoke.py

pack:
	dotnet pack src/FScript.Numerics/FScript.Numerics.csproj -c Release -o .out/packages $(if $(version),-p:Version=$(version))
	dotnet pack src/FScript.Numerics.Extensions/FScript.Numerics.Extensions.csproj -c Release -o .out/packages $(if $(version),-p:Version=$(version))
	python3 .github/scripts/verify_packages.py $(version)

verify-changelog:
	python3 .github/scripts/changelog.py

release-prepare:
	python3 .github/scripts/release.py "$(version)" "$(dryrun)"
