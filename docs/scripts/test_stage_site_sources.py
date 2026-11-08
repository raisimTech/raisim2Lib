"""Tests for docs/cmake/StageSiteSources.cmake on a fake raisim2Lib tree.

The staged tree links to rsc/, examples/ and scripts/ of the repository, so
the important property is that restaging and deleting the staged tree never
delete anything behind those links.

Run with ``python3 -m unittest discover -s docs/scripts``; needs ``cmake``.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'cmake', 'StageSiteSources.cmake')
_CMAKE = shutil.which('cmake')


@unittest.skipIf(_CMAKE is None, 'cmake is not installed')
class StageSiteSourcesTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = os.path.join(self._tmp.name, 'raisim2Lib')
        self.docs = os.path.join(self.repo, 'docs')
        self.stage = os.path.join(self._tmp.name, 'build', 'site-src')
        for path in (
                'CMakeLists.txt', 'rsc/docs/image/canary.png',
                'examples/src/canary.cpp', 'scripts/canary.py',
                'docs/index.md', 'docs/sections/Gone.md',
                'docs/sections/Kept.md', 'docs/.vitepress/config.mts',
                'docs/.vitepress/theme/index.ts', 'docs/public/favicon.png',
                'docs/scripts/check_links.py',
                # Not staged: generated, installed or cached files.
                'docs/api/index.md', 'docs/node_modules/vitepress/x.js',
                'docs/.vitepress/cache/deps.js', 'docs/.vitepress/.temp/a.js',
                'docs/.vitepress/dist/index.html', 'docs/.venv/bin/python',
                'docs/_build/old.html', 'docs/scripts/__pycache__/a.pyc',
                'docs/.api.new-0123/index.md'):
            self.write(path, path)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, path, text):
        target = os.path.join(self.repo, *path.split('/'))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w', encoding='utf-8') as stream:
            stream.write(text)

    def stage_sources(self):
        return subprocess.run(
            [_CMAKE, '-DDOCS_SOURCE_DIR=' + self.docs,
             '-DREPO_ROOT=' + self.repo, '-DSTAGE_DIR=' + self.stage,
             '-P', _SCRIPT], capture_output=True, text=True, check=False)

    def staged_files(self):
        """Files below the stage, without descending into the links."""
        found = []
        for root, dirs, files in os.walk(self.stage):
            dirs[:] = [d for d in dirs
                       if not os.path.islink(os.path.join(root, d))]
            for name in files:
                path = os.path.relpath(os.path.join(root, name), self.stage)
                found.append(path.replace(os.sep, '/'))
        return sorted(found)

    def assert_canaries(self):
        for path in ('rsc/docs/image/canary.png', 'examples/src/canary.cpp',
                     'scripts/canary.py'):
            with open(os.path.join(self.repo, *path.split('/')),
                      encoding='utf-8') as stream:
                self.assertEqual(stream.read(), path)

    def test_restaging_keeps_the_linked_trees_and_drops_deleted_pages(self):
        for _ in range(2):
            result = self.stage_sources()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assert_canaries()
        self.assertEqual(self.staged_files(), [
            'CMakeLists.txt', 'docs/.vitepress/config.mts',
            'docs/.vitepress/theme/index.ts', 'docs/index.md',
            'docs/public/favicon.png', 'docs/scripts/check_links.py',
            'docs/sections/Gone.md', 'docs/sections/Kept.md'])
        # The relative paths of the pages reach the repository through links
        # (directory junctions on Windows).
        is_junction = getattr(os.path, 'isjunction', lambda path: False)
        for name in ('rsc', 'examples', 'scripts'):
            link = os.path.join(self.stage, name)
            self.assertTrue(os.path.islink(link) or is_junction(link), link)
        with open(os.path.join(self.stage, 'docs', '..', 'rsc', 'docs',
                               'image', 'canary.png'),
                  encoding='utf-8') as stream:
            self.assertEqual(stream.read(), 'rsc/docs/image/canary.png')

        os.remove(os.path.join(self.docs, 'sections', 'Gone.md'))
        result = self.stage_sources()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('docs/sections/Gone.md', self.staged_files())
        self.assertIn('docs/sections/Kept.md', self.staged_files())
        self.assert_canaries()

    def test_deleting_the_stage_keeps_the_linked_trees(self):
        result = self.stage_sources()
        self.assertEqual(result.returncode, 0, result.stderr)
        subprocess.run([_CMAKE, '-E', 'rm', '-rf', self.stage], check=True)
        self.assertFalse(os.path.lexists(self.stage))
        self.assert_canaries()

    def test_a_directory_where_a_link_belongs_is_not_deleted(self):
        os.makedirs(os.path.join(self.stage, 'rsc'))
        canary = os.path.join(self.stage, 'rsc', 'mine.txt')
        with open(canary, 'w', encoding='utf-8') as stream:
            stream.write('mine')
        result = self.stage_sources()
        self.assertNotEqual(result.returncode, 0)
        # CMake wraps the message.
        self.assertIn('remove the path by hand',
                      ' '.join(result.stderr.split()))
        self.assertTrue(os.path.isfile(canary))


if __name__ == '__main__':
    unittest.main()
