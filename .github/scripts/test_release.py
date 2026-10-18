"""Exercise first-release and subsequent-release planning without touching the real repo."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parent

class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        subprocess.run(['git', 'init', '-q', '-b', 'main', self.root], check=True)
        self.git('config', 'user.email', 'tests@example.invalid')
        self.git('config', 'user.name', 'Release tests')
        (self.root / 'CHANGELOG.md').write_text('# Changelog\n\n## [Unreleased]\n\n- Initial numerics.\n')
        shutil.copy(SCRIPTS / 'release.py', self.root)
        (self.root / 'Directory.Build.props').write_text('<Project><PropertyGroup><Version>0.0.0</Version></PropertyGroup></Project>')
        self.commit()
        self.bootstrap = self.git('rev-parse', 'HEAD').strip()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, text=True, stderr=subprocess.STDOUT)

    def commit(self):
        self.git('add', '.')
        self.git('commit', '-qm', 'chore: test release inputs')

    def preview(self, version):
        return subprocess.run(['python3', 'release.py', version, 'true'], cwd=self.root, text=True, capture_output=True)

    def test_bootstrap_compare_and_no_mutation(self):
        result = self.preview('0.1.0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'{self.bootstrap}...0.1.0', result.stdout)
        self.assertEqual(self.git('status', '--porcelain'), '')
        self.assertEqual(self.git('tag'), '')

    def test_subsequent_compare(self):
        self.git('tag', '0.1.0')
        result = self.preview('0.1.1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('0.1.0...0.1.1', result.stdout)

    def test_invalid_versions_and_dirty_tree(self):
        for version in ['0.1.0-beta', '0.0.9', '0.1.1']:
            self.assertNotEqual(self.preview(version).returncode, 0)
        (self.root / 'dirty').write_text('pending')
        self.assertNotEqual(self.preview('0.1.0').returncode, 0)

    def test_prepare_updates_shared_version_notes_commit_and_tag(self):
        # Fake only the expensive validation commands: exercise the real Git/file mutations.
        bin_dir = self.root / 'tools'
        bin_dir.mkdir()
        make = bin_dir / 'make'
        make.write_text('#!/bin/sh\necho "$@" >> gates.log\n')
        make.chmod(0o755)
        (self.root / '.gitignore').write_text('gates.log\n')
        self.commit()
        environment = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'])
        result = subprocess.run(['python3','release.py','0.1.0'], cwd=self.root, env=environment, text=True, capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(self.git('log','-1','--format=%s').strip(),'chore(release): 0.1.0')
        self.assertEqual(self.git('cat-file','-t','0.1.0').strip(),'tag')
        self.assertIn('<Version>0.1.0</Version>',(self.root/'Directory.Build.props').read_text())
        self.assertEqual((self.root/'gates.log').read_text().splitlines(),[
            f'{target} version=0.1.0' for target in ['build','test','smoke-tests','pack','verify-changelog']])
        self.assertEqual(self.git('status','--porcelain'),'')
        command = ['python3',str(SCRIPTS/'notes.py'),'0.1.0']
        notes = subprocess.check_output(command,cwd=self.root,text=True)
        self.assertIn(f'{self.bootstrap}...0.1.0',notes)
        self.assertNotEqual(self.preview('0.1.0').returncode,0)

    def test_release_notes_require_bullets_and_compare(self):
        (self.root / 'CHANGELOG.md').write_text('# Changelog\n\n## [Unreleased]\n\n## [0.1.0]\n\n- Numerics.\n')
        command = ['python3', str(SCRIPTS / 'notes.py'), '0.1.0']
        self.assertNotEqual(subprocess.run(command, cwd=self.root, capture_output=True).returncode, 0)
        with (self.root / 'CHANGELOG.md').open('a') as stream:
            stream.write('\n**Full Changelog**: https://github.com/MagnusOpera/FScript.Numerics/compare/bootstrap...0.1.0\n')
        result = subprocess.run(command, cwd=self.root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('- Numerics.', result.stdout)

if __name__ == '__main__':
    unittest.main()
