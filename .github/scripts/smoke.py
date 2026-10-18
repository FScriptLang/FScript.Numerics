"""Run every example using the pinned, published CLI and beside-only discovery."""
from pathlib import Path
import subprocess
root = Path(__file__).resolve().parents[2]
subprocess.run(['dotnet', 'tool', 'restore'], cwd=root, check=True)
for script in sorted((root / 'examples').glob('*.fss')):
    result = subprocess.run(['dotnet','tool','run','fscript','--',str(script)], cwd=root,
                            text=True, capture_output=True, check=True)
    expected = script.with_suffix('.expected').read_text().strip()
    assert result.stdout.strip() == expected, (script.name, result.stdout, result.stderr, expected)
    print(f'{script.name}: passed')
