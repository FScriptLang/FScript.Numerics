"""Missing-native diagnosis and lazy descriptor discovery in a fresh process."""
from pathlib import Path
import subprocess,shutil
folder=Path('.out/native-failure').resolve()
shutil.rmtree(folder,ignore_errors=True)
subprocess.run(['dotnet','publish','tools/BackendProbe','-c','Release','-o',str(folder),'--nologo'],check=True)
subprocess.run(['dotnet',str(folder/'BackendProbe.dll')],check=True)
for file in folder.rglob('*'):
    if file.is_file() and file.name in ('libopenblas.dll','libopenblas.dylib','libopenblas.so'):file.unlink()
result=subprocess.run(['dotnet',str(folder/'BackendProbe.dll'),'--expect-failure'],capture_output=True,text=True,timeout=30)
assert result.returncode==0,(result.stdout,result.stderr)
assert 'Registration without native loading' in result.stdout and 'Cannot initialize LP64 OpenBLAS' in result.stdout,result.stdout
print('Fresh process: descriptors remain available; missing native assets fail with RID-specific diagnostics.')
