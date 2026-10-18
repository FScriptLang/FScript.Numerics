"""Verify the exact native binaries selected by the pinned vendor package."""
import hashlib,json,subprocess,struct
from pathlib import Path
cache=Path(subprocess.check_output(['dotnet','nuget','locals','global-packages','--list'],text=True).strip().split(': ',1)[1])
root=cache/'blassharp.openblas'/'0.3.0'/'runtimes'
expected=json.loads(Path('docs/native/asset-sha256.json').read_text())
for name,digest in expected.items():
    path=root/name
    assert path.exists(),path
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual==digest,(name,actual,digest)
    data=path.read_bytes()
    if name.endswith('.so'): assert data[:4]==b'\x7fELF' and struct.unpack_from('<H',data,18)[0]==62, name
    elif name.endswith('.dll'): assert struct.unpack_from('<H',data,struct.unpack_from('<I',data,0x3c)[0]+4)[0]==0x8664,name
    elif name.endswith('.dylib'): assert struct.unpack_from('<I',data,4)[0]==0x100000c,name
    print(f'{name}: pinned SHA-256 and actual machine type verified')
