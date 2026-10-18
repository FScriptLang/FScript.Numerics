from pathlib import Path
import subprocess,re,sys
version=sys.argv[1]; dryrun=len(sys.argv)>2 and sys.argv[2]=='true'
assert re.fullmatch(r'\d+\.\d+\.\d+',version), 'Expected X.Y.Z'
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(), 'Commit changes first'
assert subprocess.run(['git','rev-parse','--verify',f'refs/tags/{version}'],capture_output=True).returncode!=0,'Tag exists'
p=Path('CHANGELOG.md');text=p.read_text()
assert f'## [{version}]' not in text
match=re.search(r'^## \[Unreleased\]\n(.*?)(?=^## \[|\Z)',text,re.M|re.S)
assert match and re.search(r'^- .+',match[1],re.M),'Unreleased needs a bullet'
tags=subprocess.check_output(['git','tag','--list'],text=True).splitlines()
tags=sorted((t for t in tags if re.fullmatch(r'\d+\.\d+\.\d+',t)),key=lambda t:tuple(map(int,t.split('.'))))
assert tags or version == '0.1.0', 'Initial release must be 0.1.0'
assert not tags or tuple(map(int,version.split('.')))>tuple(map(int,tags[-1].split('.'))),'Release version must increase'
previous=tags[-1] if tags else subprocess.check_output(['git','rev-list','--max-parents=0','HEAD'],text=True).strip()
link=f'**Full Changelog**: https://github.com/MagnusOpera/FScript.Numerics/compare/{previous}...{version}'
body=f'## [Unreleased]\n\n## [{version}]\n\n{match[1].strip()}\n\n{link}\n\n'
if dryrun:
    print(f'Would prepare {version}; {link}');sys.exit(0)
# Validate before changing release metadata or creating any commit/tag.
for target in ['build','test','smoke-tests','pack','package-smoke-tests','verify-changelog']:
    subprocess.run(['make',target,f'version={version}'],check=True)
p.write_text(text[:match.start()]+body+text[match.end():])
props=Path('Directory.Build.props')
props.write_text(re.sub(r'(<Version[^>]*>)[^<]+(</Version>)', lambda m:m[1]+version+m[2], props.read_text()))
subprocess.run(['git','add','CHANGELOG.md','Directory.Build.props'],check=True)
subprocess.run(['git','commit','-m',f'chore(release): {version}'],check=True)
subprocess.run(['git','tag','-a',version,'-m',f'Release {version}'],check=True)
print('Push commit/tag together with git push origin main --follow-tags; publish the CI-created draft only after checking its notes.')
