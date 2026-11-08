"""Tests for generate_api_docs.py.

The fixture in testdata/api_xml is a small hand-written Doxygen XML tree:
raisim::World (members, overloads, operators, an enum, member groups and most
description markup), its base raisim::Object, the case-colliding classes
raisim::Thing and raisim::thing, a template specialization, the nested
namespace raisim::sub, the raisin namespace, classes of the global namespace
(OpenGLMesh and three whose names clash in the api/ root: Aux, Index and
Raisim), and compounds that must be skipped (a private nested class, a class
located in a source file, a detail namespace, an empty namespace and a class
of the foreign namespace contact). The fixture in testdata/api_xml_collisions
holds compounds whose page paths are equal (raisim::Spec< int > and
raisim::Spec< int * >, raisim::index and the namespace page) and a class with
two anonymous unions, as Doxygen 1.15 writes them.

Run with ``python3 -m unittest discover -s docs/scripts``.
"""

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_api_docs as gen  # pylint: disable=wrong-import-position

_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'testdata', 'api_xml')
_COLLISION_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'testdata', 'api_xml_collisions')


def _fenced_code_removed(text):
    return re.sub(r'^(`{3,}).*?^\1$', '', text, flags=re.M | re.S)


class GeneratedPagesTest(unittest.TestCase):
    """Generates the fixture once and checks the pages."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = os.path.join(cls._tmp.name, 'api')
        cls.generator = gen.generate(_FIXTURE, cls.out)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def read(self, path):
        with open(os.path.join(self.out, *path.split('/')),
                  encoding='utf-8') as stream:
            return stream.read()

    def member(self, page, anchor):
        """Returns the text of one member, from its heading to the next."""
        text = self.read(page)
        match = re.search(r'^### [^\n]*\{#%s\}$.*?(?=^##|\Z)' %
                          re.escape(anchor), text, re.M | re.S)
        self.assertIsNotNone(match, 'no member #%s in %s' % (anchor, page))
        return match.group(0)

    def test_writes_exactly_the_public_pages(self):
        # SrcOnly (source file), World::Secret (private), raisim::detail,
        # raisim::empty and contact::Detector get no page.
        self.assertEqual(sorted(self.generator.files), [
            'Aux-2.md', 'Index-2.md', 'OpenGLMesh.md', 'OpenGLMesh/Stats.md',
            'Raisim-2/Inner.md', 'Raisim.md', 'index.md', 'raisim/Object.md',
            'raisim/Prod_op-int.md', 'raisim/Thing.md', 'raisim/World.md',
            'raisim/World/Checkpoint.md', 'raisim/index.md',
            'raisim/sub/Scene.md', 'raisim/sub/index.md',
            'raisim/thing-2.md', 'raisin/index.md', 'sidebar.json'])
        written = []
        for root, _, files in os.walk(self.out):
            for name in files:
                path = os.path.relpath(os.path.join(root, name), self.out)
                written.append(path.replace(os.sep, '/'))
        self.assertEqual(sorted(written), sorted(
            list(self.generator.files) + [gen.MANIFEST_NAME]))

    def test_class_page_header(self):
        page = self.read('raisim/World.md')
        self.assertTrue(page.startswith(
            '---\ntitle: "raisim\\\\:\\\\:World"\neditLink: false\n'
            'outline: [2, 3]\n---\n\n# raisim::World {#page-title}\n'))
        self.assertIn('`class` \u00b7 `#include <raisim/World.hpp>`', page)
        # Doxygen lists the base twice; the page names it once.
        self.assertIn('**Inherits:** [raisim::Object](Object.md)\n', page)
        self.assertIn('**Derived by:** [raisim::World](World.md)',
                      self.read('raisim/Object.md'))
        self.assertIn('`abstract class`', self.read('raisim/Object.md'))

    def test_sections_are_in_order(self):
        ids = re.findall(r'^## .*\{#([a-z-]+)\}$', self.read('raisim/World.md'),
                         re.M)
        self.assertEqual(ids, [
            'doc-quick-start', 'nested-classes', 'public-types',
            'constructors-and-destructor', 'public-member-functions',
            'static-member-functions', 'public-attributes',
            'inherited-members'])

    def test_overloads_share_one_heading(self):
        page = self.read('raisim/World.md')
        self.assertEqual(page.count('{#load}'), 1)
        load = self.member('raisim/World.md', 'load')
        self.assertIn('```cpp\nvoid load(const std::string &path)\n```', load)
        self.assertIn('```cpp\nvoid load(int index) const noexcept\n```', load)

    def test_constructors_destructor_and_operators(self):
        self.assertIn('explicit World(const std::string &configFile)',
                      self.member('raisim/World.md', 'world'))
        self.assertIn('### \\~World {#destructor}',
                      self.read('raisim/World.md'))
        self.assertIn('bool operator==(const World &other) const',
                      self.member('raisim/World.md', 'operator-eq'))
        self.assertIn('virtual double operator()(int i)',
                      self.member('raisim/World.md', 'operator-call'))

    def test_badges(self):
        self.assertIn('### create <Badge type="info" text="static" /> '
                      '<Badge type="warning" text="deprecated" /> {#create}',
                      self.read('raisim/World.md'))
        # Only one overload of load() is deprecated: no badge, a container.
        load = self.member('raisim/World.md', 'load')
        self.assertNotIn('Badge', load)
        self.assertIn('::: warning Deprecated\nUse the path overload.\n:::',
                      load)

    def test_member_keeps_its_id_over_same_named_type(self):
        page = self.read('raisim/World.md')
        self.assertIn('### mode {#mode}', page)
        self.assertIn('### Mode {#mode-2}', page)
        self.assertEqual(self.generator.anchor_renames,
                         [('raisim/World.md', 'mode', 'mode-2')])
        # A reference to an enumerator links the enum.
        self.assertIn('[EXACT](#mode-2)', self.member('raisim/World.md',
                                                      'mode'))

    def test_private_protected_and_friend_members_are_skipped(self):
        page = self.read('raisim/World.md') + self.read('raisim/Object.md')
        for name in ('internalStep', 'secret\\_', 'WorldFriend',
                     'updateCollision', 'Secret'):
            self.assertNotIn(name, page)

    def test_undocumented_member_keeps_only_its_signature(self):
        self.assertEqual(
            self.member('raisim/World.md', 'operator-eq'),
            '### operator== {#operator-eq}\n\n'
            '```cpp\nbool operator==(const World &other) const\n```\n\n')

    def test_inherited_members_link_the_base_page(self):
        page = self.read('raisim/World.md')
        self.assertIn('**From [raisim::Object](Object.md):** '
                      '[getName](Object.md#getname), '
                      '[setName](Object.md#setname)\n', page)
        self.assertNotIn('Object.md#object', page)

    def test_nested_class_list(self):
        self.assertIn('- `struct` [Checkpoint](World/Checkpoint.md) \u2014 '
                      'A saved state.', self.read('raisim/World.md'))

    def test_description_markup(self):
        page = self.read('raisim/World.md')
        self.assertIn('## Quick start {#doc-quick-start}\n\n```cpp\n'
                      'auto world = raisim::World::create();\n```', page)
        load = self.member('raisim/World.md', 'load')
        for expected in (
                '**bold** and *emphasized* items',
                'Energy is $E = m c^2$ and\n\n$$\n\\sum_i x_i\n$$\n\n',
                'Read [the site](https://raisim.com) and '
                '[Object::getName()](Object.md#getname).',
                '- first point\n- second point\n\n  ```python\n'
                '  print( 1)\n  ```',
                '1. one\n2. two',
                '**Parameters**\n\n| Parameter | Description |\n',
                '**Returns** nothing',
                '::: info Note\nLoading is slow.\n:::',
                '::: warning Warning\nNot thread-safe.\n:::'):
            self.assertIn(expected, load)
        # Section titles inside member descriptions are not headings.
        self.assertIn('**Usage**\n\nCall it.',
                      self.member('raisim/World.md',
                                  'addobjectwithaverylongname'))
        self.assertIn('***\n\n**Diagnostics**\n\n### counters {#counters}',
                      page)

    def test_long_signatures_wrap_one_parameter_per_line(self):
        self.assertIn(
            '```cpp\ntemplate <typename T>\n'
            'T *addObjectWithAVeryLongName(\n'
            '    const std::vector<std::string> &names,\n'
            '    double firstValue,\n'
            '    double secondValue)\n```',
            self.member('raisim/World.md', 'addobjectwithaverylongname'))

    def test_escaping(self):
        load = self.member('raisim/World.md', 'load')
        self.assertIn('Costs \\$5 per call for a `std::vector<T>`', load)
        self.assertIn('Literal &lt;T&gt; and &#123;&#123; name }} stay text',
                      load)
        self.assertIn('so do \\*stars\\*, \\_under\\_ and \\[brackets\\]',
                      load)
        self.assertIn('| `path` *(in)* | file, as `a\\|b` or a\\|b |', load)
        enum = self.member('raisim/World.md', 'mode-2')
        self.assertIn('| `FAST` | `0` | Fast, see '
                      '<code>&#123;&#123; speed &#125;&#125;</code>. |', enum)
        self.assertIn('| `EXACT` |  | Exact \\| slow. |', enum)
        self.assertIn('### operator&lt;&lt; {#operator-shl}',
                      self.read('raisim/index.md'))
        self.assertIn('# raisim::Prod\\_op&lt; int &gt;',
                      self.read('raisim/Prod_op-int.md'))

    def test_no_page_has_vue_template_syntax_outside_code(self):
        for path in self.generator.files:
            if not path.endswith('.md'):
                continue
            # The front matter is YAML, not part of the Vue template.
            text = re.sub(r'\A---\n.*?\n---\n', '', self.read(path),
                          flags=re.S)
            text = _fenced_code_removed(text)
            text = re.sub(r'(`+).*?\1', '', text)
            text = re.sub(r'<code>.*?</code>|<Badge [^>]*/>|<br>', '', text)
            self.assertNotIn('{{', text, path)
            self.assertNotIn('<', text, path)

    def test_every_heading_has_a_unique_explicit_id(self):
        # VitePress stops the build when an explicit id repeats an id it
        # derived from an earlier heading (OpenGLMesh's title and its
        # constructor), so every heading carries an explicit, unique id.
        for path in self.generator.files:
            if not path.endswith('.md'):
                continue
            text = _fenced_code_removed(self.read(path))
            headings = re.findall(r'^#{1,6} .*$', text, re.M)
            ids = [re.search(r' \{#([a-z0-9_-]+)\}$', h) for h in headings]
            self.assertNotIn(None, ids, path)
            ids = [match.group(1) for match in ids]
            self.assertEqual(len(ids), len(set(ids)), path)
        self.assertIn('### OpenGLMesh {#openglmesh}',
                      self.read('OpenGLMesh.md'))

    def test_page_titles_are_set_in_front_matter(self):
        # Without it VitePress takes the escaped heading as the browser
        # title, which then reads "Prod_op&lt; int &gt;".
        self.assertTrue(self.read('raisim/Prod_op-int.md').startswith(
            '---\ntitle: "raisim\\\\:\\\\:Prod\\\\_op\\\\< int \\\\>"\n'))
        for path in self.generator.files:
            if path.endswith('.md') and path != 'index.md':
                self.assertIn('\ntitle: "', self.read(path), path)

    def test_template_specialization(self):
        self.assertIn('```cpp\ntemplate <>\nstruct Prod_op<int>\n```',
                      self.read('raisim/Prod_op-int.md'))

    def test_nested_namespace(self):
        page = self.read('raisim/sub/index.md')
        self.assertIn('# raisim::sub {#page-title}\n\n`namespace`\n\n'
                      'Scene files.', page)
        self.assertIn('- `struct` [Scene](Scene.md) \u2014 A parsed scene.',
                      page)
        self.assertEqual(
            self.member('raisim/sub/index.md', 'load'),
            '### load {#load}\n\n`#include <raisim/sub/Loader.hpp>`\n\n'
            '```cpp\nScene load(const std::string &file)\n```\n\n'
            'Parses a scene file into a [Scene](Scene.md) for a '
            '[World](../World.md).\n')
        self.assertIn('- [raisim::sub](sub/index.md) \u2014 Scene files.',
                      self.read('raisim/index.md'))

    def test_free_functions_enums_aliases_and_variables(self):
        page = self.read('raisim/index.md')
        enum = self.member('raisim/index.md', 'motorsaturation')
        self.assertIn('```cpp\nenum MotorSaturation\n```', enum)
        # No enumerator is documented, so there is no Description column.
        self.assertIn('| Enumerator | Value |\n| --- | --- |\n'
                      '| `NONE` | `0` |\n| `CLIPPED` |  |', enum)
        self.assertEqual(page.count('{#motortorquebounds}'), 1)
        bounds = self.member('raisim/index.md', 'motortorquebounds')
        self.assertIn('`#include <raisim/object/ArticulatedSystem/'
                      'MotorOperatingRegion.hpp>`', bounds)
        self.assertIn('double motorTorqueBounds(double speed)\n', bounds)
        self.assertIn('double motorTorqueBounds(double speed, double voltage)',
                      bounds)
        self.assertIn('[CLIPPED](#motorsaturation)', bounds)
        self.assertIn('template <size_t n>\nusing Vec = Mat<n, 1>',
                      self.member('raisim/index.md', 'vec'))
        self.assertIn('static constexpr int kFormatVersion',
                      self.member('raisim/index.md', 'kformatversion'))
        self.assertIn('[raisim::World](../raisim/World.md)',
                      self.read('raisin/index.md'))

    def test_case_collision_is_renamed_and_reported(self):
        # In the api/ root: Aux is a Windows device name, Index.md clashes
        # with the landing page, and the directory of Raisim::Inner clashes
        # with the raisim/ namespace directory, which keeps its name.
        self.assertEqual(self.generator.collisions, [
            ('Aux', 'Aux.md', 'Aux-2.md'), ('Index', 'Index.md', 'Index-2.md'),
            ('Raisim::Inner', 'Raisim/Inner.md', 'Raisim-2/Inner.md'),
            ('raisim::thing', 'raisim/thing.md', 'raisim/thing-2.md')])
        self.assertIn('# raisim::Thing {#page-title}\n',
                      self.read('raisim/Thing.md'))
        self.assertIn('# raisim::thing {#page-title}\n',
                      self.read('raisim/thing-2.md'))
        self.assertIn('# Index {#page-title}\n', self.read('Index-2.md'))
        self.assertIn('# Raisim::Inner {#page-title}\n',
                      self.read('Raisim-2/Inner.md'))
        self.assertIn('- `struct` [Inner](Raisim-2/Inner.md)',
                      self.read('Raisim.md'))
        for root, dirs, files in os.walk(self.out):
            names = [name.lower() for name in dirs + files]
            self.assertEqual(len(names), len(set(names)), root)

    def test_landing_page(self):
        page = self.read('index.md')
        self.assertIn('# API reference', page)
        self.assertIn('- [`raisim`](raisim/index.md) \u2014 ', page)
        self.assertIn('- [`raisin`](raisin/index.md) \u2014 ', page)
        self.assertIn('### Core {#raisim-core}\n\n'
                      '- `class` [Thing](raisim/Thing.md) \u2014 '
                      'Upper-case thing.\n'
                      '- `class` [thing](raisim/thing-2.md) \u2014 '
                      'Lower-case thing.\n'
                      '- `class` [World](raisim/World.md) \u2014 '
                      'The simulation world.', page)
        self.assertIn('- [raisim::sub](raisim/sub/index.md) \u2014 '
                      'Scene files.\n'
                      '  - [Scene](raisim/sub/Scene.md) \u2014 '
                      'A parsed scene.', page)
        self.assertIn('- [Global namespace](#ns-global) \u2014 ', page)
        self.assertIn('## Global namespace {#ns-global}\n\n'
                      '- `struct` [Aux](Aux-2.md) \u2014 '
                      'A Windows device name.\n'
                      '- `struct` [Index](Index-2.md) \u2014 '
                      'Clashes with the landing page.\n'
                      '- `class` [OpenGLMesh](OpenGLMesh.md) \u2014 '
                      'A GPU mesh.\n'
                      '- `struct` [Raisim](Raisim.md) \u2014 '
                      'Clashes with the raisim directory.\n', page)

    def test_global_namespace_class(self):
        page = self.read('OpenGLMesh.md')
        self.assertIn('# OpenGLMesh {#page-title}\n\n'
                      '`class` \u00b7 `#include <rayrai/OpenGLMesh.hpp>`\n\n'
                      '**Inherits:** std::enable\\_shared\\_from\\_this'
                      '&lt; OpenGLMesh &gt;\n', page)
        self.assertIn('- `struct` [Stats](OpenGLMesh/Stats.md) \u2014 '
                      'Draw statistics.', page)
        self.assertIn('[raisin::rsceneVisuals()]'
                      '(raisin/index.md#rscenevisuals)',
                      self.member('OpenGLMesh.md', 'draw'))
        self.assertIn('# OpenGLMesh::Stats {#page-title}\n',
                      self.read('OpenGLMesh/Stats.md'))

    def test_sidebar(self):
        sidebar = json.loads(self.read('sidebar.json'))
        self.assertEqual(sidebar[0], {'text': 'API reference',
                                      'link': '/api/index.md'})
        self.assertEqual([group['text'] for group in sidebar[1:]],
                         ['raisim', 'raisin', 'Global namespace'])
        global_group = sidebar[3]
        self.assertEqual([item['text'] for item in global_group['items']],
                         ['Aux', 'Index', 'OpenGLMesh', 'Raisim'])
        self.assertEqual(global_group['items'][2]['items'],
                         [{'text': 'Stats',
                           'link': '/api/OpenGLMesh/Stats.md'}])
        raisim = sidebar[1]
        self.assertFalse(raisim['collapsed'])
        self.assertEqual([item['text'] for item in raisim['items']],
                         ['Namespace raisim', 'Core', 'Objects', 'Math',
                          'Namespaces'])
        for group in raisim['items'][1:]:
            self.assertTrue(group['collapsed'])
        core = raisim['items'][1]['items']
        self.assertEqual(core[2], {
            'text': 'World', 'link': '/api/raisim/World.md',
            'collapsed': True,
            'items': [{'text': 'Checkpoint',
                       'link': '/api/raisim/World/Checkpoint.md'}]})
        self.assertEqual(raisim['items'][3]['items'][0]['text'],
                         'Prod_op&lt; int &gt;')

        links = []

        def collect(items):
            for item in items:
                if 'link' in item:
                    links.append(item['link'])
                collect(item.get('items', []))

        collect(sidebar)
        pages = sorted('/api/' + path for path in self.generator.files
                       if path.endswith('.md'))
        self.assertEqual(sorted(links), pages)

    def test_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            again = gen.generate(_FIXTURE, os.path.join(tmp, 'api'))
            self.assertEqual(again.files, self.generator.files)


class CollisionFixtureTest(unittest.TestCase):
    """Pages with equal paths, and anonymous unions."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = os.path.join(cls._tmp.name, 'api')
        cls.generator = gen.generate(_COLLISION_FIXTURE, cls.out)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def read(self, path):
        with open(os.path.join(self.out, *path.split('/')),
                  encoding='utf-8') as stream:
            return stream.read()

    def test_equal_paths_get_pages_of_their_own(self):
        self.assertEqual(sorted(self.generator.files), [
            'index.md', 'raisim/Row.md', 'raisim/Spec-int-2.md',
            'raisim/Spec-int-3.md', 'raisim/Spec-int.md',
            'raisim/index-2.md', 'raisim/index.md', 'sidebar.json'])
        titles = {}
        for path in self.generator.files:
            if path.endswith('.md') and path != 'index.md':
                titles[path] = re.search(r'^# (.*) \{#page-title\}$',
                                         self.read(path), re.M).group(1)
        self.assertEqual(titles, {
            'raisim/index.md': 'raisim',
            'raisim/index-2.md': 'raisim::index',
            'raisim/Row.md': 'raisim::Row',
            'raisim/Spec-int.md': 'raisim::Spec&lt; int \\* &gt;',
            'raisim/Spec-int-2.md': 'raisim::Spec&lt; int, 2 &gt;',
            'raisim/Spec-int-3.md': 'raisim::Spec&lt; int &gt;'})
        self.assertEqual(self.generator.collisions, [
            ('raisim::Spec< int >', 'raisim/Spec-int.md',
             'raisim/Spec-int-3.md'),
            ('raisim::index', 'raisim/index.md', 'raisim/index-2.md')])
        page = self.read('raisim/index.md')
        self.assertIn('- `struct` [index](index-2.md) \u2014 ', page)
        self.assertIn('- `struct` [Spec&lt; int &gt;](Spec-int-3.md) \u2014 ',
                      page)
        sidebar = json.loads(self.read('sidebar.json'))
        links = [item['link'] for item in sidebar[1]['items'][1]['items']]
        self.assertEqual(sorted(links), [
            '/api/raisim/Row.md', '/api/raisim/Spec-int-2.md',
            '/api/raisim/Spec-int-3.md', '/api/raisim/Spec-int.md',
            '/api/raisim/index-2.md'])

    def test_anonymous_unions_get_a_heading_each(self):
        page = self.read('raisim/Row.md')
        self.assertNotRegex(page, r'(?m)^#+ +\{')
        self.assertNotIn('union raisim::Row', page)
        note = ('An anonymous union: its members are listed with the other '
                'attributes and share storage.')
        self.assertIn('### state {#state}\n\n```cpp\nint state\n```\n\n'
                      '### (anonymous union) {#anonymous-union}\n\n' + note +
                      '\n\nSoil rows reuse the storage of the mass.\n\n'
                      '### pointId {#pointid}', page)
        self.assertTrue(page.endswith(
            '### (anonymous union) {#anonymous-union-2}\n\n' + note + '\n'))


class HelperTest(unittest.TestCase):
    """Unit tests of the naming helpers."""

    def test_member_anchor(self):
        cases = {
            'getName': 'getname', '~World': 'destructor',
            'operator==': 'operator-eq', 'operator()': 'operator-call',
            'operator[]': 'operator-subscript', 'operator<<': 'operator-shl',
            'operator bool': 'operator-bool',
            'operator new[]': 'operator-new-array',
            'operator const Vec< 3 > &': 'operator-const-vec-3',
            'operatorCount': 'operatorcount', 'get_frame': 'get_frame',
            'foo< int >': 'foo-int',
        }
        for name, anchor in cases.items():
            self.assertEqual(gen.member_anchor(name), anchor, name)

    def test_split_qualified_name(self):
        self.assertEqual(
            gen.split_qualified_name(
                'raisim::Prod_op< Mat< 4, 4 >, std::vector< int > >::Inner'),
            ['raisim', 'Prod_op< Mat< 4, 4 >, std::vector< int > >', 'Inner'])

    def test_hidden_namespaces(self):
        for name in ('detail', 'internal', 'track_detail',
                     'rayrai_internal_shaders', '@0', ''):
            self.assertTrue(gen.is_hidden_namespace_component(name), name)
        for name in ('rscene', 'tcp_viewer', 'details_view'):
            self.assertFalse(gen.is_hidden_namespace_component(name), name)

    def test_resolve_case_collisions(self):
        paths, renamed = gen.resolve_path_collisions(
            ['ns/Foo/index.md', 'ns/foo/index.md', 'ns/Foo.md', 'ns/aux.md',
             'Index.md'], reserved=['index.md'])
        self.assertEqual(paths, [
            'ns/Foo/index.md', 'ns/foo-2/index.md', 'ns/Foo.md',
            'ns/aux-2.md', 'Index-2.md'])
        self.assertEqual(renamed, [(1, 'ns/foo-2/index.md'),
                                   (3, 'ns/aux-2.md'), (4, 'Index-2.md')])

    def test_identical_paths_get_files_of_their_own(self):
        # Spec<int> and Spec<int*> both become Spec-int; the suffix skips
        # Spec-int-2, the page of Spec<int, 2>. A class index plans the
        # namespace page; equal directories are one directory.
        paths, renamed = gen.resolve_path_collisions(
            ['ns/index.md', 'ns/Spec-int.md', 'ns/Spec-int.md',
             'ns/Spec-int-2.md', 'ns/index.md', 'ns/Spec-int/Inner.md',
             'ns/Spec-int/Inner.md', 'index.md'],
            reserved=['index.md'], preferred=['ns/', 'ns/index.md'])
        self.assertEqual(paths, [
            'ns/index.md', 'ns/Spec-int.md', 'ns/Spec-int-3.md',
            'ns/Spec-int-2.md', 'ns/index-2.md', 'ns/Spec-int/Inner.md',
            'ns/Spec-int/Inner-2.md', 'index-2.md'])
        self.assertEqual([index for index, _ in renamed], [2, 4, 6, 7])
        self.assertEqual(len(set(path.lower() for path in paths)), len(paths))

    def test_preferred_entries_keep_their_names(self):
        # A namespace directory and page win over a class directory and a
        # class page that sort before them.
        paths, _ = gen.resolve_path_collisions(
            ['ns/Contact/Item.md', 'ns/contact/index.md', 'ns/Index.md',
             'ns/index.md'],
            preferred=['ns/contact/', 'ns/contact/index.md', 'ns/index.md'])
        self.assertEqual(paths, [
            'ns/Contact-2/Item.md', 'ns/contact/index.md', 'ns/Index-2.md',
            'ns/index.md'])

    def test_front_matter_title_shows_the_name(self):
        # VitePress parses the title as inline Markdown; removing the
        # backslash escapes is what it does to the text.
        for name in ('raisim::Prod_op< Mat< 4, 4 >, int * >', 'Foo<int>',
                     'a_b_*c* `d` $e$ {#f} [g](h) \\i & j'):
            line = gen.front_matter_title(name)
            self.assertTrue(line.startswith('title: "'), line)
            value = json.loads(line[len('title: '):])
            self.assertEqual(re.sub(r'\\(.)', r'\1', value), name)
            self.assertNotIn('\\&', value)

    def test_split_alphabetically(self):
        names = ['%s%d' % (letter, i) for letter in 'ABCDEFGH'
                 for i in range(5)]
        classes = [gen.Compound(name, 'class', 'ns::' + name)
                   for name in names]
        split = gen._split_alphabetically  # pylint: disable=protected-access
        groups = split('Renderer', classes)
        self.assertEqual([(label, len(members)) for label, members in groups],
                         [('Renderer A\u2013D', 20),
                          ('Renderer E\u2013H', 20)])
        self.assertEqual([c for _, members in groups for c in members],
                         classes)

    def test_code_span(self):
        self.assertEqual(gen.code_span('a`b'), '``a`b``')
        self.assertEqual(gen.code_span('{{x}}'),
                         '<code>&#123;&#123;x&#125;&#125;</code>')


class OutputDirectoryTest(unittest.TestCase):
    """The generator only deletes a directory that it wrote."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.out = os.path.join(self.root, 'api')

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, path, text='x'):
        target = os.path.join(self.out, path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w', encoding='utf-8') as stream:
            stream.write(text)

    def test_refuses_a_directory_not_named_api(self):
        out = os.path.join(self.root, 'docs')
        with self.assertRaises(gen.GeneratorError):
            gen.generate(_FIXTURE, out)
        self.assertFalse(os.path.exists(out))

    def test_refuses_a_directory_without_manifest(self):
        self.write('notes.md')
        with self.assertRaises(gen.GeneratorError):
            gen.generate(_FIXTURE, self.out)
        self.assertTrue(os.path.isfile(os.path.join(self.out, 'notes.md')))

    def test_refuses_a_file_it_did_not_write(self):
        gen.generate(_FIXTURE, self.out)
        self.write('raisim/extra.md')
        with self.assertRaises(gen.GeneratorError):
            gen.generate(_FIXTURE, self.out)
        self.assertTrue(os.path.isfile(os.path.join(self.out, 'raisim',
                                                    'extra.md')))
        self.assertTrue(os.path.isfile(os.path.join(self.out, 'index.md')))

    def test_regenerates_its_own_output(self):
        gen.generate(_FIXTURE, self.out)
        self.write('raisim/.DS_Store')
        os.remove(os.path.join(self.out, 'raisim', 'World.md'))
        gen.generate(_FIXTURE, self.out)
        self.assertTrue(os.path.isfile(os.path.join(self.out, 'raisim',
                                                    'World.md')))
        self.assertFalse(os.path.exists(os.path.join(self.out, 'raisim',
                                                     '.DS_Store')))

    def test_accepts_an_empty_directory(self):
        os.makedirs(self.out)
        gen.generate(_FIXTURE, self.out)
        self.assertTrue(os.path.isfile(os.path.join(self.out, 'index.md')))

    @unittest.skipIf(not hasattr(os, 'symlink'), 'no symlinks')
    def test_refuses_a_symlink(self):
        target = os.path.join(self.root, 'elsewhere')
        os.makedirs(target)
        try:
            os.symlink(target, self.out)
        except OSError:
            self.skipTest('cannot create symlinks')
        with self.assertRaises(gen.GeneratorError):
            gen.generate(_FIXTURE, self.out)
        self.assertTrue(os.path.isdir(target))

    def snapshot(self):
        """Every file of the output with its content."""
        files = {}
        for root, _, names in os.walk(self.out):
            for name in names:
                path = os.path.join(root, name)
                with open(path, encoding='utf-8') as stream:
                    files[os.path.relpath(path, self.out)] = stream.read()
        return files

    def siblings(self):
        return sorted(name for name in os.listdir(self.root) if name != 'api')

    def sibling(self, name, files, listed=None):
        """A directory next to the output, as old as a stale one.

        Its manifest lists ``listed``; None writes no manifest.
        """
        path = os.path.join(self.root, name)
        os.makedirs(path)
        if listed is not None:
            with open(os.path.join(path, gen.MANIFEST_NAME), 'w',
                      encoding='utf-8') as stream:
                stream.write('# manifest\n' +
                             ''.join(f + '\n' for f in listed))
        for file in files:
            os.makedirs(os.path.dirname(os.path.join(path, file)),
                        exist_ok=True)
            with open(os.path.join(path, file), 'w',
                      encoding='utf-8') as stream:
                stream.write('x')
        old = time.time() - gen.STALE_SECONDS - 60
        os.utime(path, (old, old))
        return path

    @staticmethod
    def open_failing_after(writes, error):
        """An open() that raises ``error`` on the ``writes``-th write."""
        opened = []

        def fake_open(path, mode='r', *args, **kwargs):
            if 'w' in mode:
                opened.append(path)
                if len(opened) == writes:
                    raise error
            return open(path, mode, *args, **kwargs)
        return fake_open

    def test_interrupted_write_keeps_the_previous_output(self):
        for error, raised in ((KeyboardInterrupt(), KeyboardInterrupt),
                              (OSError(28, 'No space left on device'),
                               gen.GeneratorError)):
            with self.subTest(error=error):
                gen.generate(_FIXTURE, self.out)
                before = self.snapshot()
                with mock.patch.object(gen, 'open', self.open_failing_after(
                        6, error), create=True):
                    with self.assertRaises(raised):
                        gen.generate(_COLLISION_FIXTURE, self.out)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(self.siblings(), [])
        # Without a previous output, an interrupted run writes no output.
        os.rename(self.out, os.path.join(self.root, 'kept'))
        with mock.patch.object(gen, 'open', self.open_failing_after(
                6, OSError(28, 'No space left on device')), create=True):
            with self.assertRaises(gen.GeneratorError):
                gen.generate(_FIXTURE, self.out)
        self.assertEqual(self.siblings(), ['kept'])

    def test_killed_run_leaves_a_sibling_that_a_later_run_deletes(self):
        gen.generate(_FIXTURE, self.out)
        before = self.snapshot()
        # A killed process deletes nothing.
        with mock.patch.object(gen, 'open', self.open_failing_after(
                6, OSError(28, 'No space left on device')), create=True), \
                mock.patch.object(gen, 'remove_generated_dir'):
            with self.assertRaises(gen.GeneratorError):
                gen.generate(_COLLISION_FIXTURE, self.out)
        self.assertEqual(self.snapshot(), before)
        [left] = self.siblings()
        self.assertTrue(left.startswith(gen.STAGING_PREFIX), left)
        # A young sibling may belong to a run in progress.
        gen.generate(_COLLISION_FIXTURE, self.out)
        self.assertEqual(self.siblings(), [left])
        old = time.time() - gen.STALE_SECONDS - 60
        os.utime(os.path.join(self.root, left), (old, old))
        gen.generate(_FIXTURE, self.out)
        self.assertEqual(self.siblings(), [])
        self.assertEqual(self.snapshot(), before)

    def test_only_stale_siblings_of_its_own_are_deleted(self):
        gen.generate(_FIXTURE, self.out)
        # An interrupted write: the manifest lists more than was written.
        self.sibling(gen.STAGING_PREFIX + 'partial', ['raisim/World.md'],
                     ['raisim/World.md', 'raisim/Object.md'])
        self.sibling(gen.RETIRED_PREFIX + 'previous', ['index.md'],
                     ['index.md'])
        foreign = self.sibling(gen.STAGING_PREFIX + 'notes', ['notes.md'])
        unrelated = self.sibling('.api.backup', ['index.md'])
        expected = [os.path.basename(foreign), '.api.backup']
        target = self.sibling('target', ['keep.md'], ['keep.md'])
        link = os.path.join(self.root, gen.STAGING_PREFIX + 'link')
        try:
            os.symlink(target, link, target_is_directory=True)
            expected += [os.path.basename(link)]
        except OSError:
            pass  # No symlink privilege (Windows).
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = gen.main(['--xml', _FIXTURE, '--out', self.out])
        self.assertEqual(status, 0)
        self.assertEqual(self.siblings(), sorted(expected + ['target']))
        for path in (os.path.join(foreign, 'notes.md'),
                     os.path.join(unrelated, 'index.md'),
                     os.path.join(target, 'keep.md'),
                     os.path.join(target, gen.MANIFEST_NAME)):
            self.assertTrue(os.path.isfile(path), path)
        self.assertIn('warning: kept %s: the generator did not write notes.md'
                      % foreign, stdout.getvalue())

    def test_failed_swap_restores_the_previous_output(self):
        # On Windows a rename fails while another process has a file open.
        gen.generate(_FIXTURE, self.out)
        before = self.snapshot()
        rename = os.rename

        def failing_rename(source, target):
            if os.path.basename(source).startswith(gen.STAGING_PREFIX):
                raise PermissionError(13, 'in use', source)
            rename(source, target)
        with mock.patch.object(gen.os, 'rename', failing_rename):
            with self.assertRaises(gen.GeneratorError):
                gen.generate(_COLLISION_FIXTURE, self.out)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.siblings(), [])

    def test_command_line_reports_collisions(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = gen.main(['--xml', _FIXTURE, '--out', self.out])
        self.assertEqual(status, 0)
        self.assertIn('path collision: raisim::thing (raisim/thing.md) is '
                      'written as raisim/thing-2.md', stdout.getvalue())
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = gen.main(['--xml', self.root, '--out', self.out])
        self.assertEqual(status, 1)
        self.assertIn('no Doxygen index.xml', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
