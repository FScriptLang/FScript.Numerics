from pathlib import Path
import re,sys
version=sys.argv[1]
text=Path('CHANGELOG.md').read_text()
match=re.search(r'^## \['+re.escape(version)+r'\]\n(.*?)(?=^## \[|\Z)',text,re.M|re.S)
assert match, f'Missing release section {version}'
body=match[1].strip()
assert re.search(r'^- .+',body,re.M), 'Release needs a bullet'
assert re.search(r'^\*\*Full Changelog\*\*: https://github\.com/MagnusOpera/FScript\.Numerics/compare/[^\s]+\.\.\.'+re.escape(version)+r'$', body, re.M), 'Missing compare link for this version'
assert body.splitlines()[-1].startswith('**Full Changelog**:'), 'Release section must end with its compare link'
print(body)
