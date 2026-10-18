from pathlib import Path
import os, re, subprocess
text = Path('CHANGELOG.md').read_text()
assert re.search(r'^## \[Unreleased\]$', text, re.M), 'Missing top Unreleased section'
head = subprocess.check_output(['git','log','-1','--format=%s'], text=True).strip()
if re.fullmatch(r'chore\(release\): \d+\.\d+\.\d+', head):
    assert re.search(r'^## \[\d+\.\d+\.\d+\]\n', text, re.M)
else:
    section = text.split('## [Unreleased]',1)[1].split('\n## [',1)[0]
    assert re.search(r'^- .+',section,re.M), 'Unreleased requires a short bullet'
base = os.environ.get('GITHUB_BASE_REF')
if base:
    subprocess.run(['git','fetch','origin',base],check=True)
    revision=f'origin/{base}...HEAD'
else:
    revision='HEAD~1...HEAD'
if subprocess.run(['git','rev-parse','--verify','HEAD~1'],capture_output=True).returncode==0:
    files=subprocess.check_output(['git','diff','--name-only',revision],text=True).splitlines()
    # A dirty preflight checks the proposed changes as well.
    files+=subprocess.check_output(['git','diff','--name-only','HEAD'],text=True).splitlines()
    assert not files or 'CHANGELOG.md' in files, 'Every commit must update CHANGELOG.md'
print('Changelog gate passed.')
