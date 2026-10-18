"""Restore the actual local NuGet artifacts through the published FScript host."""
import sys
from pathlib import Path
import json,os,shutil,subprocess,xml.sax.saxutils,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'.out/package-smoke'
# Always discard manifest-local restore state so rebuilt 0.1.0 artifacts cannot reuse stale assemblies.
shutil.rmtree(folder,ignore_errors=True);folder.mkdir(parents=True)
version=os.environ.get('NUMERICS_VERSION') or ET.parse(ROOT/'Directory.Build.props').findtext('./PropertyGroup/Version')
(folder/'NuGet.Config').write_text(f'''<configuration><packageSources><clear/><add key="numerics-local" value="{xml.sax.saxutils.escape(str(ROOT/'.out/packages'))}"/><add key="nuget.org" value="https://api.nuget.org/v3/index.json"/></packageSources></configuration>''')
(folder/'fscript.toml').write_text(f'[[runtime.packages]]\nname = "FScript.Numerics.Extensions"\nversion = "{version}"\n')
# Isolate NuGet's global extraction cache too: local packages intentionally retain one development version.
env=os.environ.copy();env['NUGET_PACKAGES']=str(folder/'packages')
subprocess.run(['dotnet','tool','restore'],cwd=ROOT,check=True)
for example in sorted((ROOT/'examples').glob('*.fss')):
    script=folder/example.name;shutil.copyfile(example,script)
    run=subprocess.run(['dotnet','tool','run','fscript','--',str(script)],cwd=ROOT,env=env,text=True,capture_output=True,timeout=180)
    assert run.returncode==0,(example.name,run.stdout,run.stderr)
    assert run.stdout.strip()==example.with_suffix('.expected').read_text().strip(),(example.name,run.stdout)
    print(f'NuGet -> CLI -> {example.name}: passed',flush=True)
# Use the same restored environment for LSP and DAP; no duplicate TOML or native resolution implementation.
env['NUMERICS_SMOKE_FOLDER']=str(folder)
subprocess.run([sys.executable,str(ROOT/'.github/scripts/lsp_smoke.py')],cwd=ROOT,env=env,check=True)
subprocess.run(['node',str(ROOT/'.github/scripts/dap_smoke.mjs')],cwd=ROOT,env=env,check=True)
