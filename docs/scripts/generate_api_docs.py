#!/usr/bin/env python3
"""Generates the VitePress API reference pages from Doxygen XML.

Doxygen (docs/Doxyfile.in) describes the installed public headers of RaiSim
(``raisim/include/raisim``) and rayrai (``rayrai/include/rayrai``) as XML. This
script turns that XML into one Markdown page per class and per namespace, a
landing page and the sidebar that docs/.vitepress/config.mts loads::

    python3 docs/scripts/generate_api_docs.py \\
        --xml build-docs/docs/docs/doxygen/xml --out docs/api

The output directory is replaced on every run. The pages are written to a
new sibling directory (STAGING_PREFIX), which then takes the place of the
previous output by two renames, so an interrupted run leaves the previous
output or the new one, never a mix. To protect hand-written files, the script
only replaces a directory named ``api`` that holds nothing but the files
listed in the manifest of its previous run (MANIFEST_NAME) and
operating-system metadata such as ``.DS_Store``. Siblings that an interrupted
run left behind are deleted by a later run once they are STALE_SECONDS old,
under the same check.

Page scheme (the prose pages link to these paths):

* A class, struct or union ``ns::A::B`` is written to ``api/ns/A/B.md``.
* A class of the global namespace ``A::B`` is written to ``api/A/B.md``.
* A namespace ``ns::sub`` is written to ``api/ns/sub/index.md``. It lists the
  namespace's classes and documents its free functions, enums, type aliases
  and variables.
* ``api/index.md`` is the landing page and ``api/sidebar.json`` the sidebar.

File names keep the characters ``[A-Za-z0-9_]`` of a name; any other run of
characters (the template arguments of a specialization, for example) becomes
``-``. The page title keeps the real name. Entries of one directory whose
names are equal (``Spec<int>`` and ``Spec<int*>`` both become ``Spec-int``,
``struct raisim::index`` would take the namespace's ``index.md``) or differ
only in case (macOS and Windows file systems are case-insensitive) are
disambiguated: one entry keeps its name and each other one gets the first
``-2``, ``-3``, ... suffix that no other entry of the directory uses. The
generator's own files (``index.md``, ``sidebar.json``) and namespace
directories and pages keep their names first, then the entry first in
case-sensitive sorted order; of compounds with equal paths, a namespace, then
the first by qualified name. A name that Windows reserves for a device
(``aux``, ``con``, ...) always gets a suffix. Every rename is reported
because it moves a page that prose may link to.

Every class and namespace page sets its browser title in the front matter,
because VitePress would take the HTML-escaped heading (``Prod_op&lt;...``)
as the title.

Anchors: every member heading has an explicit id, ``### load {#load}``.
Overloads share one heading. The id is the member name in lower case, with
these exceptions:

* An operator becomes ``operator-<word>``: ``operator==`` -> ``operator-eq``,
  ``operator()`` -> ``operator-call``, ``operator<<`` -> ``operator-shl``
  (OPERATOR_ANCHOR_WORDS lists them all). A conversion operator spells its
  type: ``operator bool`` -> ``operator-bool``.
* A destructor becomes ``destructor``.
* Doxygen lists the members of an anonymous union or struct with the other
  attributes and adds an unnamed entry for the union itself. That entry gets
  its own heading, ``(anonymous union)`` with the id ``anonymous-union``.
* Any other character outside ``[a-z0-9_]`` collapses to ``-``.
* When two members of one page get the same id (``Mode`` and ``mode``), a
  function or attribute keeps it and an enum, type alias or constructor gets
  a ``-2`` suffix; between members of the same rank the later declaration
  gets the suffix. Every suffix is reported.

Every other heading has an explicit id too, and it always contains a hyphen,
so it never takes an id that a member needs: the page title
(``{#page-title}``), sections (``## Public member functions
{#public-member-functions}``) and headings inside descriptions (``doc-...``).

Selection rules:

* Classes, structs, unions and namespaces inside the namespaces of
  TOP_LEVEL_NAMESPACES, and classes of the global namespace (rayrai's
  ``OpenGLMesh``, ``Shader``, ...), whose location is a public header: a
  header file below an ``include/`` directory. Other namespaces (the contact
  engine's ``contact``, ``raisim_math``) are skipped.
* A namespace component that is anonymous or has a ``detail`` or ``internal``
  token (``detail``, ``track_detail``, ``rayrai_internal_shaders``) hides the
  namespace and everything in it. Nested namespaces without public content
  are skipped.
* Non-public nested classes, private members and friend declarations are
  skipped. Protected members are skipped as well: no RaiSim or rayrai class is
  designed for user subclassing (the abstract bases Object, Sensor,
  LengthConstraint and Chart are only subclassed inside the library), so
  their protected members are engine internals.
* Inherited members are not repeated; the class page links them to the page
  of the base class that declares them.
"""

import argparse
import collections
import html
import html.entities
import json
import os
import posixpath
import re
import secrets
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

# Documented top-level namespaces and the one-line summary shown on the landing
# page (and on the namespace page when the headers carry no description).
TOP_LEVEL_NAMESPACES = collections.OrderedDict([
    ('raisim', 'The RaiSim physics engine: worlds, objects, articulated '
               'systems, contacts, sensors and the visualization server.'),
    ('raisin', 'The rayrai renderer (the namespace is spelled raisin): '
               'render windows, cameras, materials, lights, visuals and the '
               'TCP viewer.'),
])

# Sidebar and landing-page category of a class, keyed by the directory of its
# header relative to the include root. The order of first appearance is the
# display order; a directory missing here forms a group named after itself.
CATEGORY_LABELS = collections.OrderedDict([
    ('raisim', 'Core'),
    ('raisim/object', 'Objects'),
    ('raisim/object/singleBodies', 'Objects'),
    ('raisim/object/ArticulatedSystem', 'Articulated systems'),
    ('raisim/object/terrain', 'Terrain'),
    ('raisim/object/deformable', 'Deformable objects'),
    ('raisim/object/granular', 'Granular media'),
    ('raisim/constraints', 'Constraints and tendons'),
    ('raisim/contact', 'Contact and collision'),
    ('raisim/collision', 'Contact and collision'),
    ('raisim/contact_engine', 'Contact and collision'),
    ('raisim/sensors', 'Sensors'),
    ('raisim/server', 'Server and visuals'),
    ('raisim/math', 'Math'),
    ('rayrai', 'Renderer'),
    ('rayrai/shaders', 'Shaders'),
])

# Explicit id of every page title. Without it VitePress derives the id from
# the title, and the title of a global class (``OpenGLMesh``) would take the
# id of its constructor; ids with a hyphen never clash with member ids.
TITLE_ID = 'page-title'

# A class group with more entries than this is split into alphabetical blocks.
MAX_GROUP_SIZE = 30

# Written into the output directory; lists every file the generator wrote.
MANIFEST_NAME = '.generated-by-generate_api_docs'

# Files that file managers drop into directories; they may be deleted.
OS_METADATA_FILES = frozenset({'.DS_Store', 'Thumbs.db', 'desktop.ini'})

# Name prefixes of the siblings of the output directory that a run writes the
# new pages to and moves the previous output to. VitePress skips directories
# whose names start with a dot.
STAGING_PREFIX = '.api.new-'
RETIRED_PREFIX = '.api.old-'

# stat.IO_REPARSE_TAG_MOUNT_POINT, which Python defines only on Windows: the
# reparse tag of a directory junction.
_IO_REPARSE_TAG_MOUNT_POINT = 0xA0000003

# Age after which a sibling counts as left behind by an interrupted run; a
# run takes seconds, so a younger one may belong to a run in progress.
STALE_SECONDS = 3600

# Device names that Windows reserves in every directory, with any extension.
WINDOWS_RESERVED_NAMES = frozenset(
    ['con', 'prn', 'aux', 'nul'] + ['com%d' % i for i in range(1, 10)] +
    ['lpt%d' % i for i in range(1, 10)])

# Operator spellings and the words their anchors use.
OPERATOR_ANCHOR_WORDS = collections.OrderedDict([
    ('new[]', 'new-array'), ('delete[]', 'delete-array'), ('new', 'new'),
    ('delete', 'delete'), ('()', 'call'), ('[]', 'subscript'),
    ('->*', 'arrow-star'), ('->', 'arrow'), ('<=>', 'spaceship'),
    ('<<=', 'shl-assign'), ('>>=', 'shr-assign'), ('<<', 'shl'),
    ('>>', 'shr'), ('==', 'eq'), ('!=', 'ne'), ('<=', 'le'), ('>=', 'ge'),
    ('&&', 'and'), ('||', 'or'), ('++', 'inc'), ('--', 'dec'),
    ('+=', 'add-assign'), ('-=', 'sub-assign'), ('*=', 'mul-assign'),
    ('/=', 'div-assign'), ('%=', 'mod-assign'), ('&=', 'and-assign'),
    ('|=', 'or-assign'), ('^=', 'xor-assign'), ('=', 'assign'), ('<', 'lt'),
    ('>', 'gt'), ('+', 'add'), ('-', 'sub'), ('*', 'mul'), ('/', 'div'),
    ('%', 'mod'), ('&', 'bitand'), ('|', 'bitor'), ('^', 'xor'),
    ('~', 'compl'), ('!', 'not'), (',', 'comma'),
])

_HEADER_EXTENSIONS = ('.h', '.hh', '.hpp', '.hxx', '.h++')
_CLASS_KINDS = ('class', 'struct', 'union')

# Doxygen names of special characters that are not HTML entity names.
_SPECIAL_CHARACTERS = {
    'nonbreakablespace': '\u00a0', 'umlaut': '\u00a8',
    'registered': '\u00ae', 'trademark': '\u2122', 'tm': '\u2122',
    'imaginary': '\u2111', 'Aumlaut': '\u00c4', 'Eumlaut': '\u00cb',
    'Iumlaut': '\u00cf', 'Oumlaut': '\u00d6', 'Uumlaut': '\u00dc',
    'Yumlaut': '\u0178', 'aumlaut': '\u00e4', 'eumlaut': '\u00eb',
    'iumlaut': '\u00ef', 'oumlaut': '\u00f6', 'uumlaut': '\u00fc',
    'yumlaut': '\u00ff',
}

# Simple sections rendered as a bold label in front of their text.
_SIMPLESECT_LABELS = {
    'return': 'Returns', 'see': 'See also', 'pre': 'Precondition',
    'post': 'Postcondition', 'invariant': 'Invariant', 'since': 'Since',
    'version': 'Version', 'date': 'Date', 'author': 'Author',
    'authors': 'Authors', 'copyright': 'Copyright',
}

# Simple sections rendered as a VitePress custom container: (type, title).
_SIMPLESECT_CONTAINERS = {
    'note': ('info', 'Note'), 'remark': ('info', 'Remark'),
    'remarks': ('info', 'Remarks'), 'warning': ('warning', 'Warning'),
    'attention': ('warning', 'Attention'),
    'important': ('warning', 'Important'),
}

_PARAMLIST_TITLES = {
    'param': ('Parameters', 'Parameter'),
    'templateparam': ('Template parameters', 'Parameter'),
    'retval': ('Return values', 'Value'),
    'exception': ('Exceptions', 'Exception'),
}

# Elements that carry nothing for HTML output (or that the theme replaces).
_IGNORED_ELEMENTS = frozenset({
    'anchor', 'indexentry', 'internal', 'latexonly', 'manonly', 'rtfonly',
    'xmlonly', 'docbookonly', 'toclist', 'simplesectsep', 'title',
})

# Elements this generator cannot render; their occurrences are reported.
_UNSUPPORTED_ELEMENTS = frozenset({
    'image', 'dot', 'msc', 'plantuml', 'dotfile', 'mscfile', 'diafile',
    'plantumlfile', 'htmlonly',
})

# Elements that become Markdown blocks and so end the current paragraph.
_BLOCK_ELEMENTS = frozenset({
    'para', 'sect1', 'sect2', 'sect3', 'sect4', 'sect5', 'sect6',
    'simplesect', 'parameterlist', 'itemizedlist', 'orderedlist',
    'programlisting', 'verbatim', 'preformatted', 'table', 'xrefsect',
    'heading', 'hruler', 'blockquote', 'variablelist', 'details', 'parblock',
    'copydoc', 'language',
})

# Code-block language by the extension Doxygen records for \code{.ext}.
_CODE_LANGUAGES = {
    '.py': 'python', '.sh': 'bash', '.bash': 'bash', '.cmake': 'cmake',
    '.xml': 'xml', '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml',
    '.glsl': 'glsl', '.c': 'c', '.unparsed': 'text', '.txt': 'text',
    '.md': 'md',
}


class GeneratorError(Exception):
    """Raised when the input or the output directory is unusable."""


class Block(NamedTuple):
    """A rendered Markdown block; ``is_para`` marks a one-line paragraph."""
    text: str
    is_para: bool


class Member(NamedTuple):
    """A member that gets documented on a class or namespace page."""
    elem: ET.Element
    name: str
    kind: str
    section: str
    group: Optional[ET.Element]


class Compound:
    """A class, struct, union or namespace that gets a page."""

    def __init__(self, refid: str, kind: str, name: str):
        self.refid = refid
        self.kind = kind
        self.name = name
        self.parts = split_qualified_name(name)
        self.elem = None  # type: Optional[ET.Element]
        self.include = None  # type: Optional[str]
        self.parent = None  # type: Optional[Compound]
        self.classes = []  # type: List[Compound]
        self.namespaces = []  # type: List[Compound]
        self.members = []  # type: List[Member]
        self.path = ''
        self.anchors = {}  # type: Dict[str, str]
        self.registry = AnchorRegistry()

    @property
    def short_name(self) -> str:
        return self.parts[-1]

    @property
    def is_namespace(self) -> bool:
        return self.kind == 'namespace'


# ---------------------------------------------------------------------------
# Names, paths and anchors.
# ---------------------------------------------------------------------------


def split_qualified_name(name: str) -> List[str]:
    """Splits ``a::b<c::d>::e`` at the ``::`` outside of brackets."""
    parts = []
    depth = 0
    start = 0
    index = 0
    while index < len(name):
        ch = name[index]
        if ch in '<([':
            depth += 1
        elif ch in '>)]':
            depth -= 1
        elif depth == 0 and name.startswith('::', index):
            parts.append(name[start:index])
            start = index + 2
            index += 1
        index += 1
    parts.append(name[start:])
    return parts


def is_hidden_namespace_component(component: str) -> bool:
    """True for anonymous namespaces and ``detail``/``internal`` ones."""
    if not component or component.startswith('@'):
        return True
    return bool({'detail', 'internal'} & set(component.lower().split('_')))


def sanitize_path_component(name: str) -> str:
    """Turns a class or namespace name into a file-system safe name."""
    cleaned = re.sub(r'[^A-Za-z0-9_]+', '-', name).strip('-')
    return cleaned or 'unnamed'


def member_anchor(name: str) -> str:
    """Returns the heading id of a member (see the module docstring)."""
    if name.startswith('~'):
        return 'destructor'
    match = re.match(r'operator(?![A-Za-z0-9_])\s*(.*)$', name)
    if match:
        spelling = match.group(1).replace(' ', '')
        if spelling in OPERATOR_ANCHOR_WORDS:
            return 'operator-' + OPERATOR_ANCHOR_WORDS[spelling]
        slug = re.sub(r'[^a-z0-9_]+', '-', match.group(1).lower())
        return 'operator-' + (slug.strip('-') or 'unnamed')
    if not name.strip() or name.startswith('@'):
        return 'anonymous'
    slug = re.sub(r'[^a-z0-9_]+', '-', name.lower()).strip('-')
    return slug or 'member'


def heading_slug(text: str) -> str:
    """Returns a slug for a heading taken from documentation text."""
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-') or 'section'


def include_path(location_file: Optional[str]) -> Optional[str]:
    """Returns the ``#include`` path of a public header, or None.

    A public header is a header file below an ``include/`` directory; the
    path is taken relative to the last such directory.
    """
    if not location_file:
        return None
    path = location_file.replace('\\', '/')
    if '/include/' not in path or not path.lower().endswith(
            _HEADER_EXTENSIONS):
        return None
    return path.rsplit('/include/', 1)[1]


def member_include(member: Member) -> Optional[str]:
    """Returns the ``#include`` path of the header declaring a member."""
    location = member.elem.find('location')
    return include_path(location.get('file') if location is not None
                        else None)


class AnchorRegistry:
    """Hands out unique heading ids for one page."""

    def __init__(self):
        self._used = set()
        self.renamed = []  # type: List[Tuple[str, str]]

    def take(self, wanted: str) -> str:
        """Returns ``wanted``, or ``wanted-N`` when it is already taken."""
        anchor = wanted
        suffix = 2
        while anchor in self._used:
            anchor = '%s-%d' % (wanted, suffix)
            suffix += 1
        self._used.add(anchor)
        if anchor != wanted:
            self.renamed.append((wanted, anchor))
        return anchor


def _is_device_name(entry: str) -> bool:
    return entry.rstrip('/').split('.', 1)[0].lower() in WINDOWS_RESERVED_NAMES


class _DirNode:
    """A directory of planned output: subdirectories by name, and files.

    Files are (name, order) pairs. Several pages may plan the same file name;
    ``order`` (the index in the caller's list, negative for reserved files)
    tells them apart.
    """

    def __init__(self):
        self.dirs = {}  # type: Dict[str, _DirNode]
        self.files = []  # type: List[Tuple[str, int]]


def resolve_path_collisions(
        logical_paths: List[str], reserved: Iterable[str] = (),
        preferred: Iterable[str] = ()
) -> Tuple[List[str], List[Tuple[int, str]]]:
    """Gives every page a file of its own, also on case-insensitive disks.

    Two entries of a directory collide when their names are equal or differ
    only in case. In each group of colliding entries one keeps its name and
    each other one gets the first ``-N`` suffix (N = 2, 3, ...) that no entry
    of the directory uses. Equal directory paths name one directory, which
    the pages below it share; equal file paths are renamed.

    Args:
        logical_paths: POSIX paths relative to the output directory; a path
            may occur more than once.
        reserved: Paths of other files written next to the pages; they keep
            their names.
        preferred: Pages, and directories (ending with ``/``), that keep
            their names before other entries of their directory. Among equal
            names, case-sensitive sorted order decides, and among equal
            paths the earlier one in ``logical_paths``.

    Returns:
        The physical path of every logical path, in the same order, and the
        (index, physical path) pairs of the renamed ones.
    """
    reserved = sorted(set(reserved))
    preferred = set(preferred) | set(reserved)
    root = _DirNode()
    numbered = [(index - len(reserved), path)
                for index, path in enumerate(reserved)]
    for order, path in numbered + list(enumerate(logical_paths)):
        node = root
        parts = path.split('/')
        for part in parts[:-1]:
            node = node.dirs.setdefault(part, _DirNode())
        node.files.append((parts[-1], order))

    dir_renames = {}  # type: Dict[str, str]   'a/b/' -> new name of b
    file_renames = {}  # type: Dict[int, str]  order -> new file name

    def visit(node: _DirNode, prefix: str) -> None:
        # (sort key, name, is_dir, order) of every entry of the directory.
        entries = [(name.lower(), name, True, -len(reserved) - 1)
                   for name in node.dirs]
        entries += [(name.lower(), name, False, order)
                    for name, order in node.files]
        groups = collections.defaultdict(list)
        for entry in entries:
            groups[entry[0]].append(entry)
        taken = set(groups)
        for key in sorted(groups):
            ranked = sorted(groups[key], key=lambda e: (
                (prefix + e[1] + ('/' if e[2] else '')) not in preferred,
                e[1], e[3]))
            first_renamed = 0 if _is_device_name(key) else 1
            for _, name, is_dir, order in ranked[first_renamed:]:
                stem, dot, extension = ((name, '', '') if is_dir else
                                        name.partition('.'))
                suffix = 2
                candidate = '%s-2%s%s' % (stem, dot, extension)
                while candidate.lower() in taken:
                    suffix += 1
                    candidate = '%s-%d%s%s' % (stem, suffix, dot, extension)
                taken.add(candidate.lower())
                if is_dir:
                    dir_renames[prefix + name + '/'] = candidate
                else:
                    file_renames[order] = candidate
        for name, child in node.dirs.items():
            visit(child, prefix + name + '/')

    visit(root, '')
    physical_paths = []
    renamed = []
    for index, path in enumerate(logical_paths):
        parts = path.split('/')
        prefix = ''
        physical = []
        for part in parts[:-1]:
            prefix += part + '/'
            physical.append(dir_renames.get(prefix, part))
        physical.append(file_renames.get(index, parts[-1]))
        physical_paths.append('/'.join(physical))
        if physical_paths[-1] != path:
            renamed.append((index, physical_paths[-1]))
    return physical_paths, renamed


# ---------------------------------------------------------------------------
# Markdown text.
# ---------------------------------------------------------------------------


def escape_text(text: str) -> str:
    """Escapes text for Markdown that VitePress compiles as a Vue template.

    ``<`` and ``>`` would start HTML (a stray ``<T>`` breaks the Vue build),
    ``{`` could start a Vue interpolation or markdown-it-attrs, ``$`` starts
    math, and the remaining characters are Markdown syntax.
    """
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = text.replace('{', '&#123;')
    return re.sub(r'([\\`*_\[\]~$])', r'\\\1', text)


def front_matter_title(text: str) -> str:
    """Returns the ``title:`` front-matter line that shows ``text`` as is.

    VitePress parses the title as inline Markdown and keeps only its text,
    so every ASCII punctuation character is backslash-escaped, except ``&``:
    VitePress turns an escaped ``&`` into ``&amp;``, and C++ names never
    contain the ``&name;`` of an HTML entity. A JSON string is a valid YAML
    scalar.
    """
    return 'title: ' + json.dumps(re.sub(r'([!-%\'-/:-@\[-`{-~])', r'\\\1',
                                         text))


def escape_line_start(text: str) -> str:
    """Escapes a paragraph start that Markdown would read as block syntax."""
    match = re.match(r'\d+(?=[.)])', text)
    if match:
        return match.group(0) + '\\' + text[match.end():]
    if text[:1] in ('#', '+', '-', '=', ':', '|'):
        return '\\' + text
    return text


def code_span(text: str) -> str:
    """Returns inline code for ``text`` (whitespace is collapsed)."""
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return ''
    if '{{' in text:
        # Vue interpolates {{ even inside inline code; entities avoid that.
        encoded = ''.join(
            '&#%d;' % ord(ch) if ch in '{}$|*_`[]\\~' else html.escape(ch)
            for ch in text)
        return '<code>%s</code>' % encoded
    longest = max([len(run) for run in re.findall(r'`+', text)] or [0])
    fence = '`' * (longest + 1)
    pad = ' ' if text.startswith('`') or text.endswith('`') else ''
    return fence + pad + text + pad + fence


def code_block(code: str, language: str) -> str:
    """Returns a fenced code block."""
    runs = re.findall(r'^\s*(`{3,})', code, re.M)
    fence = '`' * (max([len(run) for run in runs] or [2]) + 1)
    return '%s%s\n%s\n%s' % (fence, language, code.rstrip('\n'), fence)


def container(kind: str, title: str, blocks: List[Block]) -> Block:
    """Returns a VitePress custom container (``::: kind title``)."""
    inner = '\n\n'.join(block.text for block in blocks)
    runs = re.findall(r'^\s*(:{3,})', inner, re.M)
    fence = ':' * (max([len(run) for run in runs] or [2]) + 1)
    return Block('%s %s %s\n%s\n%s' % (fence, kind, title, inner, fence),
                 False)


def table_cell(text: str) -> str:
    """Makes inline Markdown safe for a table cell."""
    return text.replace('\n', ' ').replace('|', '\\|')


def tidy_cpp(text: str) -> str:
    """Normalizes Doxygen's spacing: ``Vec< 3 >`` -> ``Vec<3>``."""
    text = re.sub(r'\s+', ' ', text).strip()
    return re.sub(r'\s+>', '>', re.sub(r'<\s+', '<', text))


def _iter_text(elem: ET.Element) -> Iterable[str]:
    """Yields the text of an element; ``<sp/>`` becomes spaces."""
    if elem.tag == 'sp':
        yield ' ' * int(elem.get('value') or 1)
    elif elem.text:
        yield elem.text
    for child in elem:
        for piece in _iter_text(child):
            yield piece
        if child.tail:
            yield child.tail


def plain_text(elem: Optional[ET.Element]) -> str:
    """Returns the text of an element with whitespace collapsed."""
    if elem is None:
        return ''
    return re.sub(r'\s+', ' ', ''.join(_iter_text(elem))).strip()


def first_sentence(text: str) -> str:
    """Returns the first sentence of plain text."""
    match = re.match(r'(.+?[.!?])(?=\s+[A-Z(`]|$)', text)
    return match.group(1) if match else text


def one_line_summary(elem: ET.Element) -> str:
    """Returns the escaped one-line summary of a compound."""
    text = plain_text(elem.find('briefdescription'))
    if not text:
        detailed = elem.find('detaileddescription')
        para = detailed.find('para') if detailed is not None else None
        text = first_sentence(plain_text(para))
    if len(text) > 200:
        text = text[:197].rsplit(' ', 1)[0] + '\u2026'
    return escape_text(text)


def _wrap_emphasis(text: str, marker: str) -> str:
    """Wraps text in emphasis markers; outer spaces stay outside."""
    stripped = text.strip()
    if not stripped:
        return text
    lead = ' ' if text[:1].isspace() else ''
    trail = ' ' if text[-1:].isspace() else ''
    return lead + marker + stripped + marker + trail


def _list_item(marker: str, blocks: List[Block]) -> str:
    """Formats blocks as one list item; continuation lines are indented."""
    text = '\n\n'.join(block.text for block in blocks) or '\u200b'
    indent = ' ' * len(marker)
    lines = text.split('\n')
    return '\n'.join([marker + lines[0]] +
                     [indent + line if line else '' for line in lines[1:]])


def _labelled(label: str, blocks: List[Block]) -> List[Block]:
    """Puts a label in front of the first paragraph of ``blocks``."""
    if blocks and blocks[0].is_para:
        return [Block('%s \u2014 %s' % (label, blocks[0].text), True)
                ] + blocks[1:]
    return [Block(label, True)] + blocks


# ---------------------------------------------------------------------------
# Description markup.
# ---------------------------------------------------------------------------


class DocRenderer:
    """Converts Doxygen description markup to VitePress Markdown.

    Args:
        generator: Resolves references and collects unsupported constructs.
        page: Path of the page being written, relative to the output root.
        anchors: The heading ids of that page.
        heading_level: Markdown level of the outermost titled section, or
            None to render section titles as bold text (member descriptions
            sit below a member heading).
    """

    def __init__(self, generator: 'ApiDocGenerator', page: str,
                 anchors: AnchorRegistry, heading_level: Optional[int]):
        self._gen = generator
        self._page = page
        self._anchors = anchors
        self._heading_level = heading_level
        self._section_depth = 0

    def render(self, *elements: Optional[ET.Element]) -> List[Block]:
        """Renders description elements (brief, detailed, ...) as blocks."""
        blocks = []
        for elem in elements:
            if elem is not None:
                blocks.extend(self._mixed(elem))
        return blocks

    def inline(self, elem: Optional[ET.Element]) -> str:
        """Renders an element as one line of inline Markdown."""
        if elem is None:
            return ''
        return ' '.join(block.text.replace('\n', ' ')
                        for block in self._mixed(elem))

    # Mixed content --------------------------------------------------------

    def _mixed(self, elem: ET.Element) -> List[Block]:
        """Renders text and children; block children end a paragraph."""
        blocks = []
        buffer = [escape_text(elem.text or '')]
        for child in elem:
            if self._is_block(child):
                self._flush(buffer, blocks)
                blocks.extend(self._block(child))
            else:
                buffer.append(self._inline(child))
            buffer.append(escape_text(child.tail or ''))
        self._flush(buffer, blocks)
        return blocks

    @staticmethod
    def _flush(buffer: List[str], blocks: List[Block]) -> None:
        text = re.sub(r'\s+', ' ', ''.join(buffer)).strip()
        del buffer[:]
        if text:
            blocks.append(Block(escape_line_start(text), True))

    def _is_block(self, elem: ET.Element) -> bool:
        if elem.tag == 'formula':
            return self._formula(elem)[1]
        return elem.tag in _BLOCK_ELEMENTS

    # Block elements -------------------------------------------------------

    def _block(self, elem: ET.Element) -> List[Block]:
        tag = elem.tag
        if tag in ('para', 'parblock', 'copydoc', 'language'):
            return self._mixed(elem)
        if re.match(r'sect\d$', tag):
            return self._section(elem)
        if tag == 'simplesect':
            return self._simplesect(elem)
        if tag == 'parameterlist':
            return self._parameter_list(elem)
        if tag in ('itemizedlist', 'orderedlist'):
            return [self._list(elem, tag == 'orderedlist')]
        if tag == 'programlisting':
            return [self._program_listing(elem)]
        if tag in ('verbatim', 'preformatted'):
            code = ''.join(_iter_text(elem)).strip('\n')
            return [Block(code_block(code, 'text'), False)]
        if tag == 'table':
            return self._table(elem)
        if tag == 'xrefsect':
            title = plain_text(elem.find('xreftitle')) or 'See also'
            deprecated = elem.get('id', '').startswith('deprecated')
            return [container('warning' if deprecated else 'info',
                              escape_text(title),
                              self.render(elem.find('xrefdescription')))]
        if tag == 'heading':
            text = self.inline(elem)
            return [Block('**%s**' % text, True)] if text else []
        if tag == 'hruler':
            return [Block('***', False)]
        if tag == 'blockquote':
            inner = '\n\n'.join(block.text for block in self._mixed(elem))
            return [Block('\n'.join('> ' + line if line else '>'
                                    for line in inner.split('\n')), False)]
        if tag == 'variablelist':
            return [self._variable_list(elem)]
        if tag == 'details':
            return [self._details(elem)]
        # A display formula.
        return [Block(self._formula(elem)[0], False)]

    def _section(self, elem: ET.Element) -> List[Block]:
        title_elem = elem.find('title')
        title = self.inline(title_elem)
        blocks = []
        if title:
            self._section_depth += 1
            if self._heading_level is None:
                blocks.append(Block('**%s**' % title, True))
            else:
                level = min(6, self._heading_level + self._section_depth - 1)
                anchor = self._anchors.take(
                    'doc-' + heading_slug(plain_text(title_elem)))
                blocks.append(Block('%s %s {#%s}' % ('#' * level, title,
                                                     anchor), False))
        blocks.extend(self._mixed(elem))
        if title:
            self._section_depth -= 1
        return blocks

    def _simplesect(self, elem: ET.Element) -> List[Block]:
        kind = elem.get('kind', '')
        if kind == 'rcs':
            return []
        blocks = self._mixed(elem)
        if kind in _SIMPLESECT_CONTAINERS:
            container_kind, title = _SIMPLESECT_CONTAINERS[kind]
            return [container(container_kind, title, blocks)]
        if kind == 'par':
            label = self.inline(elem.find('title'))
        else:
            label = _SIMPLESECT_LABELS.get(kind, kind.capitalize())
        if not label:
            return blocks
        if blocks and blocks[0].is_para:
            return [Block('**%s** %s' % (label, blocks[0].text), True)
                    ] + blocks[1:]
        return [Block('**%s**' % label, True)] + blocks

    def _parameter_list(self, elem: ET.Element) -> List[Block]:
        title, column = _PARAMLIST_TITLES.get(elem.get('kind', 'param'),
                                              _PARAMLIST_TITLES['param'])
        rows = []
        for item in elem.findall('parameteritem'):
            names = []
            for name_elem in item.iter('parametername'):
                name = code_span(plain_text(name_elem))
                direction = name_elem.get('direction')
                if direction:
                    name += ' *(%s)*' % direction.replace('inout', 'in, out')
                names.append(name)
            rows.append((', '.join(names),
                         self.render(item.find('parameterdescription'))))
        if not rows:
            return []
        blocks = [Block('**%s**' % title, True)]
        if all(block.is_para for _, desc in rows for block in desc):
            lines = ['| %s | Description |' % column, '| --- | --- |']
            for names, desc in rows:
                lines.append('| %s | %s |' % (
                    table_cell(names),
                    table_cell('<br>'.join(block.text for block in desc))))
            blocks.append(Block('\n'.join(lines), False))
        else:
            items = [_list_item('- ', _labelled(names, desc))
                     for names, desc in rows]
            blocks.append(Block('\n'.join(items), False))
        return blocks

    def _list(self, elem: ET.Element, ordered: bool) -> Block:
        items = []
        for index, item in enumerate(elem.findall('listitem'), 1):
            marker = '%d. ' % index if ordered else '- '
            items.append(_list_item(marker, self._mixed(item)))
        return Block('\n'.join(items), False)

    def _variable_list(self, elem: ET.Element) -> Block:
        items = []
        term = ''
        for child in elem:
            if child.tag == 'varlistentry':
                term = self.inline(child.find('term'))
            elif child.tag == 'listitem':
                blocks = self._mixed(child)
                if term:
                    blocks = _labelled('**%s**' % term, blocks)
                items.append(_list_item('- ', blocks))
        return Block('\n'.join(items), False)

    def _details(self, elem: ET.Element) -> Block:
        summary = elem.find('summary')
        body = ET.Element('details')
        body.text = elem.text
        body.extend([child for child in elem if child.tag != 'summary'])
        title = self.inline(summary) if summary is not None else 'Details'
        return container('details', title, self._mixed(body))

    def _program_listing(self, elem: ET.Element) -> Block:
        lines = [''.join(_iter_text(line)).rstrip()
                 for line in elem.findall('codeline')]
        extension = (elem.get('filename') or '').lower()
        if extension and not extension.startswith('.'):
            extension = os.path.splitext(extension)[1]
        language = _CODE_LANGUAGES.get(extension, 'cpp')
        return Block(code_block('\n'.join(lines).strip('\n'), language),
                     False)

    def _table(self, elem: ET.Element) -> List[Block]:
        rows = []
        for row in elem.findall('row'):
            cells = []
            for entry in row.findall('entry'):
                cell = self._mixed(entry)
                if any(not block.is_para for block in cell):
                    self._gen.note_unsupported('table cell with block content',
                                               self._page)
                cells.append(table_cell('<br>'.join(block.text
                                                    for block in cell)))
            rows.append(cells)
        if not rows:
            return []
        width = max(len(cells) for cells in rows)
        rows = [cells + [''] * (width - len(cells)) for cells in rows]
        lines = ['| %s |' % ' | '.join(rows[0]),
                 '| %s |' % ' | '.join(['---'] * width)]
        lines.extend('| %s |' % ' | '.join(cells) for cells in rows[1:])
        blocks = []
        caption = self.inline(elem.find('caption'))
        if caption:
            blocks.append(Block('**%s**' % caption, True))
        blocks.append(Block('\n'.join(lines), False))
        return blocks

    # Inline elements ------------------------------------------------------

    def _inline(self, elem: ET.Element) -> str:
        tag = elem.tag
        if tag == 'computeroutput':
            return self._code(elem)
        if tag == 'ref':
            return self._link(self._inline_content(elem), elem.get('refid'))
        if tag == 'ulink':
            url = elem.get('url', '')
            text = self._inline_content(elem) or escape_text(url)
            for char, encoded in ((' ', '%20'), ('(', '%28'), (')', '%29')):
                url = url.replace(char, encoded)
            return '[%s](%s)' % (text, url)
        if tag == 'bold':
            return _wrap_emphasis(self._inline_content(elem), '**')
        if tag == 'emphasis':
            return _wrap_emphasis(self._inline_content(elem), '*')
        if tag in ('s', 'strike', 'del'):
            return _wrap_emphasis(self._inline_content(elem), '~~')
        if tag in ('underline', 'ins', 'subscript', 'superscript', 'small'):
            html_tag = {'underline': 'ins', 'subscript': 'sub',
                        'superscript': 'sup'}.get(tag, tag)
            return '<%s>%s</%s>' % (html_tag, self._inline_content(elem),
                                    html_tag)
        if tag in ('center', 'cite', 'highlight', 'summary', 'term'):
            return self._inline_content(elem)
        if tag == 'formula':
            return self._formula(elem)[0]
        if tag == 'linebreak':
            return '<br>'
        if tag == 'sp':
            return ' '
        if tag == 'emoji':
            return elem.get('unicode') or escape_text(elem.get('name', ''))
        if tag in _SPECIAL_CHARACTERS:
            return _SPECIAL_CHARACTERS[tag]
        if (tag in html.entities.name2codepoint and len(elem) == 0 and
                not (elem.text or '').strip()):
            return escape_text(chr(html.entities.name2codepoint[tag]))
        if tag in _IGNORED_ELEMENTS:
            return ''
        self._gen.note_unsupported(tag, self._page)
        if tag in _UNSUPPORTED_ELEMENTS:
            return ''
        return self._inline_content(elem)

    def _inline_content(self, elem: ET.Element) -> str:
        """Renders the content of an inline element on one line."""
        pieces = [escape_text(elem.text or '')]
        for child in elem:
            if self._is_block(child):
                pieces.extend(' ' + block.text.replace('\n', ' ')
                              for block in self._block(child))
            else:
                pieces.append(self._inline(child))
            pieces.append(escape_text(child.tail or ''))
        return re.sub(r'\s+', ' ', ''.join(pieces))

    def _link(self, text: str, refid: Optional[str]) -> str:
        href = self._gen.href(refid, self._page)
        return '[%s](%s)' % (text, href) if href and text.strip() else text

    def _code(self, elem: ET.Element) -> str:
        """Renders ``computeroutput``; references inside become links."""
        refs = list(elem.iter('ref'))
        if len(refs) <= 1:
            span = code_span(plain_text(elem))
            return self._link(span, refs[0].get('refid')) if refs else span
        pieces = [code_span(elem.text or '')]
        for child in elem:
            span = code_span(plain_text(child))
            if child.tag == 'ref':
                span = self._link(span, child.get('refid'))
            pieces.extend([span, code_span(child.tail or '')])
        return ''.join(pieces)

    def _formula(self, elem: ET.Element) -> Tuple[str, bool]:
        """Returns (Markdown, is_display) for a formula.

        Doxygen keeps the delimiters of the source: ``$...$`` and ``\\(...\\)``
        are inline, ``\\[...\\]`` and ``\\begin{env}...`` are display math.
        """
        text = (elem.text or '').strip()
        delimiters = (('$$', '$$', True), ('$', '$', False),
                      ('\\(', '\\)', False), ('\\[', '\\]', True))
        for opening, closing, display in delimiters:
            if (text.startswith(opening) and text.endswith(closing) and
                    len(text) > len(opening) + len(closing)):
                tex = text[len(opening):-len(closing)].strip()
                return ('$$\n%s\n$$' % tex if display else '$%s$' % tex,
                        display)
        if text.startswith('\\begin'):
            return '$$\n%s\n$$' % text, True
        self._gen.note_unsupported('formula without delimiters', self._page)
        return code_span(text), False


# ---------------------------------------------------------------------------
# Signatures.
# ---------------------------------------------------------------------------


def template_line(elem: ET.Element) -> str:
    """Returns ``template <...>`` for an element with a templateparamlist."""
    params = elem.find('templateparamlist')
    if params is None:
        return ''
    rendered = []
    for param in params.findall('param'):
        text = plain_text(param.find('type'))
        declname = plain_text(param.find('declname'))
        if declname and not text.endswith(declname):
            text = '%s %s' % (text, declname)
        default = plain_text(param.find('defval'))
        if default:
            text = '%s = %s' % (text, default)
        rendered.append(tidy_cpp(text))
    return 'template <%s>' % ', '.join(rendered)


def _join_type_and_name(type_text: str, name: str) -> str:
    if not type_text:
        return name
    if type_text.endswith(('&', '*', '(')):
        return type_text + name
    return '%s %s' % (type_text, name)


def _wrap_arguments(head: str, args: str, max_width: int = 76) -> str:
    """Puts one parameter per line when the declaration is too long.

    76 characters fit the code blocks of the VitePress content column.

    The split only happens at commas outside of brackets; a parameter list
    whose brackets do not balance (a default value with ``a < b``) stays on
    one line.
    """
    if len(head) + len(args) <= max_width or not args.startswith('('):
        return head + args
    depth = 0
    close = -1
    for index, ch in enumerate(args):
        depth += {'(': 1, ')': -1}.get(ch, 0)
        if depth == 0:
            close = index
            break
    params = []
    depth = 0
    start = 1
    for index in range(1, close):
        ch = args[index]
        if ch in '(<[{':
            depth += 1
        elif ch in ')>]}':
            depth -= 1
            if depth < 0:
                return head + args
        elif ch == ',' and depth == 0:
            params.append(args[start:index].strip())
            start = index + 1
    params.append(args[start:close].strip())
    if close < 0 or depth != 0 or len(params) < 2:
        return head + args
    return '%s(\n    %s)%s' % (head, ',\n    '.join(params), args[close + 1:])


def member_signature(elem: ET.Element) -> str:
    """Builds the C++ declaration shown in a member's code block."""
    kind = elem.get('kind')
    name = elem.findtext('name') or ''
    type_text = tidy_cpp(plain_text(elem.find('type')))
    args = tidy_cpp(elem.findtext('argsstring') or '')
    args = re.sub(r'\s*=\s*(0|default|delete)$', r' = \1', args)
    template = template_line(elem)
    lines = [template] if template else []

    if kind == 'enum':
        declaration = 'enum class' if elem.get('strong') == 'yes' else 'enum'
        if not name.startswith('@'):
            declaration += ' ' + name
        if type_text:
            declaration += ' : ' + type_text
        return '\n'.join(lines + [declaration])
    if kind == 'typedef':
        if (elem.findtext('definition') or '').lstrip().startswith('using'):
            declaration = 'using %s = %s' % (name, type_text)
        else:
            declaration = 'typedef ' + _join_type_and_name(type_text,
                                                           name) + args
        return '\n'.join(lines + [declaration])

    keywords = []
    for attribute in ('static', 'constexpr', 'explicit', 'mutable'):
        if elem.get(attribute) == 'yes':
            keywords.append(attribute)
    if (elem.get('virt') in ('virtual', 'pure-virtual') and
            not re.search(r'\b(override|final)\b', args)):
        keywords.append('virtual')
    type_words = type_text.split()
    keywords = [word for word in keywords if word not in type_words]
    head = _join_type_and_name(' '.join(keywords + [type_text]).strip(), name)
    if kind == 'variable':
        declaration = head + args
        bitfield = (elem.findtext('bitfield') or '').strip()
        if bitfield:
            declaration += ' : ' + bitfield
        initializer = plain_text(elem.find('initializer'))
        if initializer and len(initializer) <= 80:
            declaration += (' ' if initializer.startswith('=') else '') + (
                initializer)
        return '\n'.join(lines + [declaration])
    return '\n'.join(lines + [_wrap_arguments(head, args)])


# ---------------------------------------------------------------------------
# The generator.
# ---------------------------------------------------------------------------


class ApiDocGenerator:
    """Reads Doxygen XML and renders the API pages in memory.

    Call load(), assign_paths() and render_pages() in that order.
    """

    def __init__(self, xml_dir: str):
        self._xml_dir = xml_dir
        self.compounds = []  # type: List[Compound]
        self._by_name = {}  # type: Dict[str, Compound]
        self._by_refid = {}  # type: Dict[str, Compound]
        # refid -> (page path, anchor) of every documented compound/member.
        self._targets = {}  # type: Dict[str, Tuple[str, Optional[str]]]
        # (qualified name, logical path, physical path) of renamed pages.
        self.collisions = []  # type: List[Tuple[str, str, str]]
        self.anchor_renames = []  # type: List[Tuple[str, str, str]]
        self.unsupported = collections.OrderedDict()
        self.files = collections.OrderedDict()  # type: Dict[str, str]
        self.warnings = []  # type: List[str]

    # Loading --------------------------------------------------------------

    def load(self) -> None:
        """Reads index.xml and the compounds of the documented namespaces."""
        index_path = os.path.join(self._xml_dir, 'index.xml')
        if not os.path.isfile(index_path):
            raise GeneratorError('no Doxygen index.xml in %s' % self._xml_dir)
        entries = ET.parse(index_path).getroot().findall('compound')
        namespaces = set(entry.findtext('name') or '' for entry in entries
                         if entry.get('kind') == 'namespace')
        candidates = []
        for entry in entries:
            kind = entry.get('kind')
            if kind not in _CLASS_KINDS and kind != 'namespace':
                continue
            compound = Compound(entry.get('refid'), kind,
                                entry.findtext('name') or '')
            if _in_documented_scope(compound, namespaces):
                candidates.append(compound)
        self._by_name = {c.name: c for c in candidates}
        for compound in candidates:
            compound.elem = self._read_compound(compound.refid)
            location = compound.elem.find('location')
            if not compound.is_namespace and location is not None:
                compound.include = include_path(location.get('file'))

        included = [c for c in candidates if self._is_included(c)]
        for compound in included:
            compound.parent = self._by_name.get('::'.join(compound.parts[:-1]))
            if compound.parent is None:
                continue
            if compound.is_namespace:
                compound.parent.namespaces.append(compound)
            else:
                compound.parent.classes.append(compound)
        for compound in included:
            compound.members = self._collect_members(compound)
        # Deepest first, drop nested namespaces without public content.
        dropped = set()
        for compound in sorted(included, key=lambda c: -len(c.parts)):
            if (compound.is_namespace and compound.parent is not None and
                    not (compound.classes or compound.members or
                         compound.namespaces)):
                compound.parent.namespaces.remove(compound)
                dropped.add(compound.name)
        self.compounds = sorted(
            (c for c in included if c.name not in dropped),
            key=lambda c: c.name)
        self._by_refid = {c.refid: c for c in self.compounds}

    def _read_compound(self, refid: str) -> ET.Element:
        path = os.path.join(self._xml_dir, refid + '.xml')
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as error:
            raise GeneratorError('cannot read %s: %s' % (path, error))
        for elem in root.iter('compounddef'):
            if elem.get('id') == refid:
                return elem
        raise GeneratorError('%s has no compounddef %s' % (path, refid))

    def _is_included(self, compound: Compound) -> bool:
        """Applies the selection rules of the module docstring."""
        if compound.is_namespace:
            return True
        if compound.elem.get('prot', 'public') != 'public':
            return False
        if compound.include is None:
            return False
        if len(compound.parts) == 1:
            return True  # A class of the global namespace.
        parent = self._by_name.get('::'.join(compound.parts[:-1]))
        return parent is not None and self._is_included(parent)

    def _collect_members(self, compound: Compound) -> List[Member]:
        """Returns the documented members in declaration order."""
        members = []
        for section in compound.elem.findall('sectiondef'):
            group = section if section.get('kind') == 'user-defined' else None
            for elem in section.findall('memberdef'):
                kind = elem.get('kind')
                if kind in ('friend', 'define'):
                    continue
                if elem.get('prot', 'public') != 'public':
                    continue
                member = Member(elem, elem.findtext('name') or '', kind,
                                _member_section(compound, elem), group)
                if member_include(member) is not None:
                    members.append(member)
        return members

    # Paths and anchors ----------------------------------------------------

    def assign_paths(self) -> None:
        """Gives every page its path and every member its anchor."""
        # Namespaces first: a namespace page keeps index.md before a class
        # named index.
        ordered = sorted(self.compounds,
                         key=lambda c: (not c.is_namespace, c.name))
        logical = []
        namespace_entries = []
        for compound in ordered:
            parts = [sanitize_path_component(p) for p in compound.parts]
            if compound.is_namespace:
                logical.append('/'.join(parts + ['index.md']))
                namespace_entries.extend(['/'.join(parts) + '/', logical[-1]])
            else:
                logical.append('/'.join(parts) + '.md')
        physical, renamed = resolve_path_collisions(
            logical, reserved=['index.md', 'sidebar.json', MANIFEST_NAME],
            preferred=namespace_entries)
        for compound, path in zip(ordered, physical):
            compound.path = path
            self._targets[compound.refid] = (compound.path, None)
        self.collisions = sorted((ordered[index].name, logical[index], path)
                                 for index, path in renamed)

        for compound in self.compounds:
            # Functions and attributes get their plain id first: prose links
            # them far more often than a same-named enum or alias (``mode``
            # versus ``Mode``) or a constructor.
            for member in sorted(_ordered_members(compound), key=lambda m: (
                    m.section in ('enums', 'types', 'constructors'))):
                key = _member_key(member)
                if key not in compound.anchors:
                    compound.anchors[key] = compound.registry.take(
                        _wanted_anchor(member))
                anchor = compound.anchors[key]
                self._targets[member.elem.get('id')] = (compound.path, anchor)
                for value in member.elem.findall('enumvalue'):
                    self._targets[value.get('id')] = (compound.path, anchor)
            for wanted, given in compound.registry.renamed:
                self.anchor_renames.append((compound.path, wanted, given))

    def href(self, refid: Optional[str], from_page: str) -> Optional[str]:
        """Returns a relative link to a documented compound or member."""
        target = self._targets.get(refid or '')
        if target is None:
            return None
        page, anchor = target
        if page == from_page:
            return '#' + anchor if anchor else None
        relative = posixpath.relpath(page, posixpath.dirname(from_page) or '.')
        return relative + ('#' + anchor if anchor else '')

    def note_unsupported(self, construct: str, page: str) -> None:
        """Records a Doxygen construct that the pages leave out."""
        pages = self.unsupported.setdefault(construct, [])
        if page not in pages:
            pages.append(page)

    # Pages ----------------------------------------------------------------

    def render_pages(self) -> Dict[str, str]:
        """Returns {path relative to the output root: file content}."""
        self.files = collections.OrderedDict()
        for compound in self.compounds:
            if compound.is_namespace:
                self.files[compound.path] = self._namespace_page(compound)
            else:
                self.files[compound.path] = self._class_page(compound)
        self.files['index.md'] = self._landing_page()
        self.files['sidebar.json'] = json.dumps(self._sidebar(),
                                                indent=2) + '\n'
        return self.files

    def _description(self, compound: Compound) -> List[str]:
        renderer = DocRenderer(self, compound.path, compound.registry, 2)
        blocks = renderer.render(compound.elem.find('briefdescription'),
                                 compound.elem.find('detaileddescription'))
        if not blocks and compound.name in TOP_LEVEL_NAMESPACES:
            blocks = [Block(escape_text(TOP_LEVEL_NAMESPACES[compound.name]),
                            True)]
        return _join_blocks(blocks)

    def _class_page(self, compound: Compound) -> str:
        elem = compound.elem
        kind = compound.kind
        if elem.get('abstract') == 'yes':
            kind = 'abstract ' + kind
        elif elem.get('final') == 'yes':
            kind = 'final ' + kind
        lines = _front_matter(bool(compound.members), compound.name)
        lines.extend([
            '# %s {#%s}' % (escape_text(compound.name), TITLE_ID), '',
            '%s \u00b7 %s' % (code_span(kind), code_span(
                '#include <%s>' % compound.include)), ''])
        relations = []
        for tag, label in (('basecompoundref', 'Inherits'),
                           ('derivedcompoundref', 'Derived by')):
            links = []
            for ref in elem.findall(tag):
                link = self._link(escape_text(ref.text or ''),
                                  ref.get('refid'), compound.path)
                if ref.get('prot', 'public') != 'public':
                    link += ' (%s)' % ref.get('prot')
                if link not in links:  # Doxygen repeats some bases.
                    links.append(link)
            if links:
                relations.append('**%s:** %s' % (label, ', '.join(links)))
        if relations:
            lines.extend([' \\\n'.join(relations), ''])
        template = template_line(elem)
        if template:
            lines.extend([code_block('%s\n%s %s' % (
                template, compound.kind, tidy_cpp(compound.short_name)),
                'cpp'), ''])
        lines.extend(self._description(compound))
        if compound.classes:
            lines.extend(['## Nested classes {#nested-classes}', ''])
            lines.extend(self._class_list(compound.classes, compound.path))
            lines.append('')
        lines.extend(self._member_sections(compound, [
            ('types', 'Public types', 'public-types'),
            ('constructors', 'Constructors & destructor',
             'constructors-and-destructor'),
            ('functions', 'Public member functions',
             'public-member-functions'),
            ('static-functions', 'Static member functions',
             'static-member-functions'),
            ('attributes', 'Public attributes', 'public-attributes'),
        ]))
        lines.extend(self._inherited_members(compound))
        return _finish(lines)

    def _namespace_page(self, compound: Compound) -> str:
        lines = _front_matter(bool(compound.members), compound.name)
        lines.extend(['# %s {#%s}' % (escape_text(compound.name), TITLE_ID),
                      '',
                      code_span('namespace'), ''])
        lines.extend(self._description(compound))
        if compound.namespaces:
            lines.extend(['## Namespaces {#nested-namespaces}', ''])
            for child in sorted(compound.namespaces, key=lambda c: c.name):
                lines.append(self._summary_item(
                    escape_text(child.name), child, compound.path))
            lines.append('')
        if compound.classes:
            lines.extend(['## Classes {#classes-and-structs}', ''])
            groups = _class_groups(compound)
            for label, classes in groups:
                if len(groups) > 1:
                    lines.extend(['**%s**' % escape_text(label), ''])
                lines.extend(self._class_list(classes, compound.path))
                lines.append('')
        lines.extend(self._member_sections(compound, [
            ('enums', 'Enumerations', 'enum-types'),
            ('types', 'Type aliases', 'type-aliases'),
            ('functions', 'Functions', 'free-functions'),
            ('variables', 'Variables', 'namespace-variables'),
        ]))
        return _finish(lines)

    def _link(self, text: str, refid: Optional[str], page: str) -> str:
        href = self.href(refid, page)
        return '[%s](%s)' % (text, href) if href else text

    def _summary_item(self, text: str, compound: Compound, page: str,
                      prefix: str = '- ') -> str:
        summary = one_line_summary(compound.elem)
        return '%s%s%s' % (prefix, self._link(text, compound.refid, page),
                           ' \u2014 ' + summary if summary else '')

    def _class_list(self, classes: List[Compound], page: str) -> List[str]:
        return [self._summary_item(escape_text(child.short_name), child, page,
                                   '- %s ' % code_span(child.kind))
                for child in sorted(classes, key=_class_sort_key)]

    def _member_sections(self, compound: Compound,
                         titles: List[Tuple[str, str, str]]) -> List[str]:
        """Renders the member sections; overloads share one heading."""
        by_key = collections.OrderedDict()  # type: Dict[str, List[Member]]
        for member in _ordered_members(compound):
            by_key.setdefault(_member_key(member), []).append(member)
        lines = []
        for section, title, anchor in titles:
            keys = [key for key, overloads in by_key.items()
                    if overloads[0].section == section]
            if not keys:
                continue
            if compound.is_namespace:
                keys.sort(key=lambda k: _member_sort_key(by_key[k][0]))
            lines.extend(['## %s {#%s}' % (title, anchor), ''])
            current_group = None
            for index, key in enumerate(keys):
                group = by_key[key][0].group
                if group is not None and group is not current_group:
                    if index > 0:
                        # Separates the label from the previous member.
                        lines.extend(['***', ''])
                    lines.extend(self._group_header(compound, group))
                current_group = group
                lines.extend(self._member(compound, key, by_key[key]))
        return lines

    def _group_header(self, compound: Compound,
                      group: ET.Element) -> List[str]:
        """Renders the title of a Doxygen member group (@name ... @{)."""
        lines = []
        header = plain_text(group.find('header'))
        if header:
            lines.extend(['**%s**' % escape_text(header), ''])
        renderer = DocRenderer(self, compound.path, compound.registry, None)
        lines.extend(_join_blocks(renderer.render(group.find('description'))))
        return lines

    def _member(self, compound: Compound, key: str,
                overloads: List[Member]) -> List[str]:
        badges = ''
        if overloads[0].section == 'static-functions':
            badges += ' <Badge type="info" text="static" />'
        if all(_is_deprecated(member.elem) for member in overloads):
            badges += ' <Badge type="warning" text="deprecated" />'
        lines = ['### %s%s {#%s}' % (_member_title(overloads[0]), badges,
                                     compound.anchors[key]), '']
        if compound.is_namespace:
            headers = sorted(set(member_include(m) for m in overloads))
            lines.extend([' '.join(code_span('#include <%s>' % header)
                                   for header in headers), ''])
        for member in overloads:
            aggregate = _anonymous_aggregate(member.elem)
            if aggregate:
                # Doxygen's declaration names the enclosing class instead.
                lines.extend([_ANONYMOUS_AGGREGATE_NOTES[aggregate], ''])
            else:
                lines.extend([code_block(member_signature(member.elem),
                                         'cpp'), ''])
            renderer = DocRenderer(self, compound.path, compound.registry,
                                   None)
            lines.extend(_join_blocks(renderer.render(
                member.elem.find('briefdescription'),
                member.elem.find('detaileddescription'))))
            if member.kind == 'enum':
                lines.extend(_enum_table(member.elem, renderer))
        return lines

    def _inherited_members(self, compound: Compound) -> List[str]:
        """One line per base class, linking the members it contributes."""
        listing = compound.elem.find('listofallmembers')
        if listing is None:
            return []
        by_base = collections.OrderedDict()  # type: Dict[str, List[str]]
        for entry in listing.findall('member'):
            base_refid = entry.get('refid', '').rsplit('_1', 1)[0]
            name = entry.findtext('name') or ''
            if (base_refid != compound.refid and
                    entry.get('prot') == 'public' and
                    name not in compound.anchors):
                names = by_base.setdefault(base_refid, [])
                if name not in names:
                    names.append(name)
        ancestors = self._ancestors(compound)
        lines = []
        for base_refid, names in sorted(by_base.items(), key=lambda item: (
                ancestors.index(item[0]) if item[0] in ancestors
                else len(ancestors))):
            base = self._by_refid.get(base_refid)
            if base is None:
                continue
            constructors = (base.short_name, '~' + base.short_name)
            names = [n for n in names
                     if n in base.anchors and n not in constructors]
            if not names:
                continue
            base_href = self.href(base.refid, compound.path)
            links = ', '.join('[%s](%s#%s)' % (escape_text(n), base_href,
                                               base.anchors[n])
                              for n in names)
            lines.extend(['**From [%s](%s):** %s' % (
                escape_text(base.name), base_href, links), ''])
        if not lines:
            return []
        return ['## Inherited members {#inherited-members}', ''] + lines

    def _ancestors(self, compound: Compound) -> List[str]:
        """Refids of the documented base classes, nearest first."""
        order = []
        queue = [compound]
        while queue:
            current = queue.pop(0)
            for ref in current.elem.findall('basecompoundref'):
                base = self._by_refid.get(ref.get('refid', ''))
                if base is not None and base.refid not in order:
                    order.append(base.refid)
                    queue.append(base)
        return order

    # Landing page and sidebar ---------------------------------------------

    def _top_level(self) -> List[Compound]:
        """The documented top-level namespaces."""
        return [c for c in self.compounds
                if c.is_namespace and len(c.parts) == 1]

    def _global_classes(self) -> List[Compound]:
        """The classes of the global namespace, without nested ones."""
        return sorted((c for c in self.compounds
                       if not c.is_namespace and len(c.parts) == 1),
                      key=_class_sort_key)

    def _landing_page(self) -> str:
        lines = ['---', 'editLink: false', 'outline: [2, 3]', '---', '',
                 '# API reference {#%s}' % TITLE_ID, '',
                 'This reference is generated from the C++ headers. It '
                 'covers these namespaces:', '']
        for top in self._top_level():
            lines.append('- [%s](%s) \u2014 %s' % (
                code_span(top.name), top.path,
                escape_text(TOP_LEVEL_NAMESPACES[top.name])))
        global_classes = self._global_classes()
        if global_classes:
            lines.append('- [Global namespace](#ns-global) \u2014 classes '
                         'that the headers declare outside any namespace.')
        lines.append('')
        for top in self._top_level():
            lines.extend([
                '## %s {#ns-%s}' % (escape_text(top.name), top.name), '',
                'The [namespace page](%s) documents the free functions, '
                'enums, type aliases and variables of %s.' % (
                    top.path, code_span(top.name)), ''])
            for label, classes in _class_groups(top):
                lines.extend(['### %s {#%s-%s}' % (
                    escape_text(label), top.name, heading_slug(label)), ''])
                lines.extend(self._class_list(classes, 'index.md'))
                lines.append('')
            nested = _nested_namespaces(top)
            if nested:
                lines.extend(['### Namespaces {#%s-namespaces}' % top.name,
                              ''])
                for namespace in nested:
                    lines.append(self._summary_item(
                        escape_text(namespace.name), namespace, 'index.md'))
                    for child in sorted(namespace.classes,
                                        key=_class_sort_key):
                        lines.append(self._summary_item(
                            escape_text(child.short_name), child, 'index.md',
                            '  - '))
                lines.append('')
        if global_classes:
            lines.extend(['## Global namespace {#ns-global}', ''])
            groups = _split_alphabetically('Classes', global_classes)
            for label, classes in groups:
                if len(groups) > 1:
                    lines.extend(['### %s {#global-%s}' % (
                        escape_text(label), heading_slug(label)), ''])
                lines.extend(self._class_list(classes, 'index.md'))
                lines.append('')
        return _finish(lines)

    def _sidebar(self) -> List[Dict]:
        sidebar = [{'text': 'API reference', 'link': '/api/index.md'}]
        for top in self._top_level():
            items = [{'text': 'Namespace %s' % top.name,
                      'link': _site(top.path)}]
            for label, classes in _class_groups(top):
                items.append(_sidebar_group(
                    label, [_sidebar_class(c) for c in classes]))
            namespaces = []
            for namespace in _nested_namespaces(top):
                item = {'text': html.escape('::'.join(namespace.parts[1:])),
                        'link': _site(namespace.path)}
                if namespace.classes:
                    item['collapsed'] = True
                    item['items'] = [
                        _sidebar_class(c)
                        for c in sorted(namespace.classes,
                                        key=_class_sort_key)]
                namespaces.append(item)
            if namespaces:
                items.append(_sidebar_group('Namespaces', namespaces))
            sidebar.append({'text': top.name, 'collapsed': False,
                            'items': items})
        global_classes = self._global_classes()
        if global_classes:
            groups = _split_alphabetically('Classes', global_classes)
            if len(groups) == 1:
                items = [_sidebar_class(c) for c in global_classes]
            else:
                items = [_sidebar_group(label, [_sidebar_class(c)
                                                for c in classes])
                         for label, classes in groups]
            sidebar.append({'text': 'Global namespace', 'collapsed': False,
                            'items': items})
        return sidebar


# ---------------------------------------------------------------------------
# Helpers of the generator.
# ---------------------------------------------------------------------------


def _in_documented_scope(compound: Compound, namespaces: Set[str]) -> bool:
    """True for compounds of TOP_LEVEL_NAMESPACES and the global namespace.

    Args:
        compound: A class or namespace from index.xml.
        namespaces: Qualified names of every namespace in index.xml; they
            tell an enclosing namespace from an enclosing class.
    """
    parts = compound.parts
    scopes = len(parts) if compound.is_namespace else len(parts) - 1
    enclosing = [parts[i] for i in range(scopes)
                 if '::'.join(parts[:i + 1]) in namespaces]
    if not enclosing:
        return not compound.is_namespace
    return (parts[0] in TOP_LEVEL_NAMESPACES and
            not any(is_hidden_namespace_component(p) for p in enclosing))


def _member_section(compound: Compound, elem: ET.Element) -> str:
    """Returns the page section of a member."""
    kind = elem.get('kind')
    if compound.is_namespace:
        return {'enum': 'enums', 'typedef': 'types',
                'function': 'functions'}.get(kind, 'variables')
    if kind in ('enum', 'typedef'):
        return 'types'
    if kind != 'function':
        return 'attributes'
    name = elem.findtext('name') or ''
    short_name = re.sub(r'<.*$', '', compound.short_name).strip()
    if name in (short_name, '~' + short_name):
        return 'constructors'
    return 'static-functions' if elem.get('static') == 'yes' else 'functions'


_SECTION_ORDER = ['enums', 'types', 'constructors', 'functions',
                  'static-functions', 'attributes', 'variables']

# Shown instead of the declaration of an anonymous union or struct.
_ANONYMOUS_AGGREGATE_NOTES = {
    'union': 'An anonymous union: its members are listed with the other '
             'attributes and share storage.',
    'struct': 'An anonymous struct: its members are listed with the other '
              'attributes.',
    'class': 'An anonymous class: its members are listed with the other '
             'attributes.',
}


def _anonymous_aggregate(elem: ET.Element) -> Optional[str]:
    """Returns ``union``, ``struct`` or ``class`` for an anonymous aggregate.

    Doxygen documents the members of an anonymous union (or struct) as
    members of the enclosing class, and the union itself as an unnamed
    variable whose type is the keyword and the name of the enclosing class.
    Returns None for every other member.
    """
    if elem.get('kind') != 'variable' or (elem.findtext('name') or '').strip():
        return None
    keyword = plain_text(elem.find('type')).split(' ', 1)[0]
    return keyword if keyword in _ANONYMOUS_AGGREGATE_NOTES else None


def _member_key(member: Member) -> str:
    """Overloads share their name as key; each unnamed member has its own."""
    if member.name.strip():
        return member.name
    return '@' + (member.elem.get('id') or '')


def _wanted_anchor(member: Member) -> str:
    aggregate = _anonymous_aggregate(member.elem)
    return 'anonymous-' + aggregate if aggregate else member_anchor(
        member.name)


def _member_title(member: Member) -> str:
    aggregate = _anonymous_aggregate(member.elem)
    if aggregate:
        return '(anonymous %s)' % aggregate
    if not member.name.strip() or member.name.startswith('@'):
        return '(anonymous)'
    return escape_text(member.name)


def _ordered_members(compound: Compound) -> List[Member]:
    """Members in page order: by section, then by declaration."""
    ranked = sorted(enumerate(compound.members),
                    key=lambda item: (_SECTION_ORDER.index(item[1].section),
                                      item[0]))
    return [member for _, member in ranked]


def _member_sort_key(member: Member) -> Tuple[str, int]:
    """Orders namespace members by header and line."""
    location = member.elem.find('location')
    return (member_include(member) or '', int(location.get('line') or 0))


def _class_sort_key(compound: Compound) -> Tuple[str, str]:
    return (compound.short_name.lower(), compound.short_name)


def _is_deprecated(elem: ET.Element) -> bool:
    return any(section.get('id', '').startswith('deprecated')
               for section in elem.iter('xrefsect'))


def _enum_table(elem: ET.Element, renderer: DocRenderer) -> List[str]:
    """Renders the enumerators of an enum as a table."""
    values = elem.findall('enumvalue')
    if not values:
        return []
    rows = []
    for value in values:
        initializer = re.sub(r'^=\s*', '',
                             plain_text(value.find('initializer')))
        blocks = renderer.render(value.find('briefdescription'),
                                 value.find('detaileddescription'))
        rows.append([code_span(value.findtext('name') or ''),
                     code_span(initializer),
                     '<br>'.join(block.text for block in blocks)])
    # Leave out the Value and Description columns when they are all empty.
    columns = [0] + [i for i in (1, 2) if any(row[i] for row in rows)]
    header = ['Enumerator', 'Value', 'Description']
    lines = ['| %s |' % ' | '.join(header[i] for i in columns),
             '| %s |' % ' | '.join(['---'] * len(columns))]
    for row in rows:
        lines.append('| %s |' % ' | '.join(table_cell(row[i])
                                           for i in columns))
    return lines + ['']


def _class_groups(namespace: Compound) -> List[Tuple[str, List[Compound]]]:
    """Groups the classes of a namespace for lists and the sidebar.

    Classes of a top-level namespace are grouped by the directory of their
    header (CATEGORY_LABELS); large groups are split alphabetically.
    """
    classes = sorted(namespace.classes, key=_class_sort_key)
    if len(namespace.parts) > 1:
        return _split_alphabetically('Classes', classes)
    labels = list(CATEGORY_LABELS.values())
    grouped = collections.OrderedDict()  # type: Dict[str, List[Compound]]
    for compound in classes:
        directory = posixpath.dirname(compound.include or '')
        grouped.setdefault(CATEGORY_LABELS.get(directory, directory),
                           []).append(compound)
    ordered = sorted(grouped.items(), key=lambda item: (
        labels.index(item[0]) if item[0] in labels else len(labels),
        item[0]))
    groups = []
    for label, members in ordered:
        groups.extend(_split_alphabetically(label, members))
    return groups


def _split_alphabetically(label: str, classes: List[Compound]) -> List[
        Tuple[str, List[Compound]]]:
    """Splits a large group into blocks of whole initial letters.

    The number of blocks is the smallest that keeps blocks near
    MAX_GROUP_SIZE; letters are then packed so that every block is close to
    the average size.
    """
    if len(classes) <= MAX_GROUP_SIZE:
        return [(label, classes)]
    by_letter = collections.OrderedDict()  # type: Dict[str, List[Compound]]
    for compound in classes:
        by_letter.setdefault(compound.short_name[:1].upper(),
                             []).append(compound)
    target = len(classes) / -(-len(classes) // MAX_GROUP_SIZE)
    blocks = []  # type: List[List[Tuple[str, List[Compound]]]]
    size = 0
    for letter, members in by_letter.items():
        grown = size + len(members)
        if blocks and abs(grown - target) <= abs(size - target):
            blocks[-1].append((letter, members))
            size = grown
        else:
            blocks.append([(letter, members)])
            size = len(members)
    groups = []
    for block in blocks:
        first, last = block[0][0], block[-1][0]
        span = first if first == last else '%s\u2013%s' % (first, last)
        groups.append(('%s %s' % (label, span),
                       [c for _, members in block for c in members]))
    return groups


def _nested_namespaces(top: Compound) -> List[Compound]:
    found = []
    stack = list(top.namespaces)
    while stack:
        namespace = stack.pop()
        found.append(namespace)
        stack.extend(namespace.namespaces)
    return sorted(found, key=lambda c: (c.name.lower(), c.name))


def _sidebar_class(compound: Compound) -> Dict:
    item = {'text': html.escape(compound.short_name),
            'link': _site(compound.path)}
    if compound.classes:
        item['collapsed'] = True
        item['items'] = [_sidebar_class(child) for child in
                         sorted(compound.classes, key=_class_sort_key)]
    return item


def _sidebar_group(label: str, items: List[Dict]) -> Dict:
    return {'text': html.escape(label), 'collapsed': True, 'items': items}


def _site(path: str) -> str:
    return '/api/' + path


def _front_matter(has_members: bool, title: str) -> List[str]:
    lines = ['---', front_matter_title(title), 'editLink: false']
    if has_members:
        lines.append('outline: [2, 3]')
    return lines + ['---', '']


def _join_blocks(blocks: List[Block]) -> List[str]:
    lines = []
    for block in blocks:
        lines.extend([block.text, ''])
    return lines


def _finish(lines: List[str]) -> str:
    return '\n'.join(lines).rstrip('\n') + '\n'


# ---------------------------------------------------------------------------
# Output directory.
# ---------------------------------------------------------------------------


def _is_link(path: str) -> bool:
    """True for a symbolic link and for a Windows directory junction."""
    if os.path.islink(path):
        return True
    try:
        tag = getattr(os.lstat(path), 'st_reparse_tag', 0)
    except OSError:
        return False
    return tag == _IO_REPARSE_TAG_MOUNT_POINT


def _foreign_files(directory: str) -> List[str]:
    """Files (and directory links) of a directory that its manifest omits.

    Operating-system metadata is never foreign. Without a manifest every
    other file is foreign.
    """
    found = []
    for root, dirs, files in os.walk(directory):
        links = [name for name in dirs
                 if _is_link(os.path.join(root, name))]
        for name in files + links:
            path = os.path.relpath(os.path.join(root, name), directory)
            found.append(path.replace(os.sep, '/'))
    written = set()
    manifest = os.path.join(directory, MANIFEST_NAME)
    if os.path.isfile(manifest):
        with open(manifest, encoding='utf-8') as stream:
            written = set(line.strip() for line in stream
                          if line.strip() and not line.startswith('#'))
    return sorted(path for path in found
                  if path != MANIFEST_NAME and path not in written and
                  posixpath.basename(path) not in OS_METADATA_FILES)


def check_output_dir(out_dir: str) -> None:
    """Checks that ``out_dir`` may be replaced.

    Raises:
        GeneratorError: If the directory is not named ``api``, is not a
            plain directory, or holds a file that the manifest of the
            previous run does not list.
    """
    if os.path.basename(os.path.normpath(out_dir)) != 'api':
        raise GeneratorError('refusing to write to %s: the output directory '
                             'must be named "api"' % out_dir)
    if not os.path.lexists(out_dir):
        return
    if _is_link(out_dir) or not os.path.isdir(out_dir):
        raise GeneratorError('refusing to replace %s: not a directory' %
                             out_dir)
    foreign = _foreign_files(out_dir)
    if not foreign:
        return
    if not os.path.isfile(os.path.join(out_dir, MANIFEST_NAME)):
        raise GeneratorError('refusing to replace %s: it has no %s, so this '
                             'generator did not write it' %
                             (out_dir, MANIFEST_NAME))
    raise GeneratorError(
        'refusing to replace %s: the generator did not write %s' %
        (out_dir, ', '.join(foreign[:10])))


def remove_generated_dir(directory: str) -> None:
    """Deletes a directory that passed _foreign_files, manifest last.

    An interrupted deletion therefore leaves a directory that a later run
    still recognizes as its own.
    """
    for entry in os.scandir(directory):
        if entry.name == MANIFEST_NAME:
            continue
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path)
        else:
            os.remove(entry.path)
    manifest = os.path.join(directory, MANIFEST_NAME)
    if os.path.lexists(manifest):
        os.remove(manifest)
    os.rmdir(directory)


def remove_stale_siblings(parent: str) -> List[str]:
    """Deletes the siblings that interrupted runs left in ``parent``.

    Only a plain directory with a sibling prefix that is STALE_SECONDS old
    and holds nothing but the files of its manifest is deleted.

    Returns:
        A warning for every left-behind sibling that is kept.
    """
    warnings = []
    for entry in sorted(os.scandir(parent), key=lambda e: e.name):
        if not entry.name.startswith((STAGING_PREFIX, RETIRED_PREFIX)):
            continue
        try:
            if _is_link(entry.path) or not entry.is_dir():
                warnings.append('kept %s: not a directory' % entry.path)
                continue
            if time.time() - entry.stat().st_mtime < STALE_SECONDS:
                continue  # It may belong to a run in progress.
            foreign = _foreign_files(entry.path)
            if foreign:
                warnings.append('kept %s: the generator did not write %s' %
                                (entry.path, ', '.join(foreign[:10])))
                continue
            remove_generated_dir(entry.path)
        except FileNotFoundError:
            continue  # Another run deleted it first.
    return warnings


def _unused_sibling(parent: str, prefix: str) -> str:
    while True:
        path = os.path.join(parent, prefix + secrets.token_hex(6))
        if not os.path.lexists(path):
            return path


def write_output(out_dir: str, files: Dict[str, str]) -> None:
    """Writes the manifest first, then the files it lists, into ``out_dir``.

    Because the manifest comes first, an interrupted write leaves only files
    that the manifest lists.
    """
    with open(os.path.join(out_dir, MANIFEST_NAME), 'w', encoding='utf-8',
              newline='\n') as stream:
        stream.write('# Written by docs/scripts/generate_api_docs.py, which '
                     'replaces this directory on every run.\n')
        for path in sorted(files):
            stream.write(path + '\n')
    for path, content in files.items():
        target = os.path.join(out_dir, *path.split('/'))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(content)


def replace_output(out_dir: str, files: Dict[str, str]) -> List[str]:
    """Writes ``files`` to a new sibling and swaps it in for ``out_dir``.

    The previous output is renamed away, the new one renamed into place and
    the previous one deleted, so ``out_dir`` holds either output in full.

    Returns:
        Warnings about left-behind siblings that were kept.

    Raises:
        GeneratorError: If ``out_dir`` may not be replaced or a file
            operation fails; ``out_dir`` then keeps the previous output.
    """
    check_output_dir(out_dir)
    parent = os.path.dirname(os.path.abspath(out_dir))
    try:
        os.makedirs(parent, exist_ok=True)
        warnings = remove_stale_siblings(parent)
        staging = _unused_sibling(parent, STAGING_PREFIX)
        os.mkdir(staging)
    except OSError as error:
        raise GeneratorError('cannot prepare %s: %s' % (out_dir, error))
    retired = None
    try:
        write_output(staging, files)
        if os.path.lexists(out_dir):
            retired = _unused_sibling(parent, RETIRED_PREFIX)
            os.rename(out_dir, retired)
            # A rename keeps the time stamp; renew it so that another run
            # does not take the retired output for a stale one meanwhile.
            os.utime(retired)
        try:
            os.rename(staging, out_dir)
        except OSError:
            if retired is not None:
                os.rename(retired, out_dir)
                retired = None
            raise
    except BaseException as error:
        try:
            remove_generated_dir(staging)
        except OSError:
            pass  # A later run deletes it; the original error matters.
        if isinstance(error, OSError):
            raise GeneratorError('cannot write %s: %s' % (out_dir, error))
        raise
    if retired is not None:
        try:
            remove_generated_dir(retired)
        except OSError as error:
            warnings.append('could not delete the previous output %s: %s; '
                            'a later run deletes it' % (retired, error))
    return warnings


def generate(xml_dir: str, out_dir: str) -> ApiDocGenerator:
    """Regenerates ``out_dir``; returns the generator for its report."""
    check_output_dir(out_dir)
    generator = ApiDocGenerator(xml_dir)
    generator.load()
    generator.assign_paths()
    files = generator.render_pages()
    generator.warnings.extend(replace_output(out_dir, files))
    return generator


def _report(generator: ApiDocGenerator, out_dir: str, verbose: bool) -> None:
    pages = sum(1 for path in generator.files if path.endswith('.md'))
    print('generate_api_docs: wrote %d pages to %s' % (pages, out_dir))
    for warning in generator.warnings:
        print('generate_api_docs: warning: %s' % warning)
    for name, logical, physical in generator.collisions:
        print('generate_api_docs: path collision: %s (%s) is written as %s' %
              (name, logical, physical))
    if verbose:
        for page, wanted, given in generator.anchor_renames:
            print('generate_api_docs: duplicate anchor in %s: #%s is written '
                  'as #%s' % (page, wanted, given))
    elif generator.anchor_renames:
        print('generate_api_docs: %d member ids got a numeric suffix because '
              'another member of the page has the same id (--verbose lists '
              'them)' % len(generator.anchor_renames))
    for construct, pages in generator.unsupported.items():
        more = ' (+%d more)' % (len(pages) - 3) if len(pages) > 3 else ''
        print('generate_api_docs: not rendered: %s in %s%s' %
              (construct, ', '.join(pages[:3]), more))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Generates the VitePress API reference from Doxygen XML.')
    parser.add_argument('--xml', required=True,
                        help='Doxygen XML directory (with index.xml)')
    parser.add_argument('--out', required=True,
                        help='output directory; it must be named "api" and '
                             'is deleted and rewritten')
    parser.add_argument('--verbose', action='store_true',
                        help='list every member id that got a suffix')
    args = parser.parse_args(argv)
    try:
        generator = generate(args.xml, args.out)
    except GeneratorError as error:
        print('generate_api_docs: error: %s' % error, file=sys.stderr)
        return 1
    _report(generator, args.out, args.verbose)
    return 0


if __name__ == '__main__':
    sys.exit(main())
