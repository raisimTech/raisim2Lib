"""Generate stubs from a built module and refine renderer return types from its inventory."""
import argparse
import ast
import json
import keyword
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module-dir', required=True)
args = parser.parse_args()
output = ROOT / 'raisimPy/typings'
env = os.environ.copy()
env['PYTHONPATH'] = str(Path(args.module_dir).resolve()) + os.pathsep + env.get('PYTHONPATH', '')
subprocess.run([sys.executable, str(ROOT / 'thirdParty/nanobind/src/stubgen.py'),
                '-m', 'raisimpy', '-m', 'raisimpy_rayrai', '-r', '-O', str(output)],
               env=env, check=True)
records = json.loads((ROOT / 'raisimPy/rayrai/api.json').read_text())


def template_args(text):
    parts, start, depth = [], 0, 0
    for index, char in enumerate(text):
        if char == '<':
            depth += 1
        elif char == '>':
            depth -= 1
        elif char == ',' and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    return parts + [text[start:].strip()]


def python_type(native):
    native = re.sub(r'^const |[ &]+$', '', native).strip()
    if native == 'std::shared_ptr<const unsigned int>':
        return 'TextureHandle'
    if native == 'void':
        return 'None'
    if native == 'bool':
        return 'bool'
    if native in ('float', 'double', 'long double'):
        return 'float'
    if native in ('int', 'unsigned int', 'unsigned long', 'unsigned long long',
                  'long', 'long long', 'short', 'unsigned short', 'unsigned char', 'char'):
        return 'int'
    if native.startswith(('std::basic_string<', 'std::basic_string_view<')) or native == 'const char *':
        return 'str'
    if native.startswith('glm::'):
        return 'numpy.ndarray'
    if native.endswith(' *'):
        return python_type(native[:-2]) + ' | None'
    for prefix in ('std::shared_ptr', 'std::optional', 'std::vector', 'std::array', 'std::span'):
        if native.startswith(prefix + '<'):
            inner = template_args(native[len(prefix) + 1:-1])[0]
            item = python_type(inner)
            if prefix == 'std::shared_ptr':
                return item
            if prefix == 'std::optional':
                return item + ' | None'
            return 'list[' + item + ']'
    if native == 'raisim::Shape::Type':
        return 'raisimpy.ShapeType'
    if native.startswith('raisin::'):
        return native.removeprefix('raisin::').replace('::', '.')
    if native.startswith('raisim::'):
        return 'raisimpy.' + native.removeprefix('raisim::').replace('::', '.')
    if native in ('Vertex', 'Texture', 'OpenGLMesh', 'Shader') or native.startswith(('OpenGLMesh::', 'Shader::')):
        return native.replace('::', '.')
    return 'object'


returns = {}
for record in records:
    if record['kind'] == 'FunctionDecl':
        methods = [{**record, 'kind': 'FunctionDecl', 'name': record['name'].split('::')[-1]}]
    else:
        methods = record.get('members', [])
    for member in methods:
        if member['kind'] == 'FieldDecl':
            result = python_type(member['canonical'])
        elif member['kind'] in ('CXXMethod', 'FunctionDecl'):
            result_native = member['resultcanonical']
            result = python_type(result_native)
            extra = []
            for argument in member.get('args', []):
                native = argument['canonical']
                base = re.sub(r'^const |[ &]+$', '', native)
                scalar_output = native.endswith(' &') and not native.startswith('const ') and (
                    base.startswith(('glm::', 'std::vector<', 'std::array<')) or
                    base in ('bool', 'int', 'float', 'double', 'unsigned int', 'unsigned long',
                             'long long', 'raisim::Shape::Type'))
                if 'basic_ostream' in native or native == 'std::basic_string<char> *':
                    extra.append('str')
                elif scalar_output:
                    extra.append(python_type(base))
            if extra:
                result = 'tuple[' + ', '.join(([result] if result_native != 'void' else []) + extra) + ']'
        else:
            continue
        owner = record['name'].removeprefix('raisin::').replace('::', '.')
        if member['kind'] in ('FunctionDecl','FunctionTemplate'):
            owner = owner.rsplit('.', 1)[0] if '.' in owner else ''
        name=member['name']+'_' if keyword.iskeyword(member['name']) else member['name']
        if owner=='RayraiGlobalAsset' and name=='getMeshes' and len(member.get('args',[]))==2:
            name='getMeshesWithCanonicalKey'
        if member['kind']=='FunctionDecl' and record['name']=='raisin::assimp::loadModel' and len(member.get('args',[]))==2:
            name='loadModelWithSourceFiles' 
        returns.setdefault((owner, name), set()).add(result)

# NumPy adapters replace native buffer/output-reference signatures.
for name in ('readRgba', 'readSceneColorBgra', 'readLinearDepth'):
    returns[('Camera', name)] = {'numpy.ndarray'}
for name in ('readSceneColorRgbaAsync', 'readLinearDepthAsync', 'getRawImage'):
    returns[('Camera', name)] = {'numpy.ndarray | None'}
returns[('RayraiWindow', 'readRgba')] = {'numpy.ndarray'}
returns[('RayraiWindow', 'setWeatherThunderCallback')] = {'None'}
returns[('RayraiWindow', 'modelHandle')] = {'ModelHandle | None'}
returns[('RayraiWindow', 'getExternalFont')] = {'Font | None'}
returns[('RayraiWindow', 'debugObjectFor')] = {'RaisimObject | None'}
returns[('assimp', 'loadModelSceneAsync')] = {'ModelLoadFuture'}

root_names = {record['name'].removeprefix('raisin::').split('::')[0]
              for record in records if record['name'].startswith('raisin::') or
              record['name'].startswith(('OpenGLMesh', 'Shader', 'Vertex', 'Texture'))}

def qualify_submodule_return(annotation):
    for name in sorted(root_names, key=len, reverse=True):
        annotation = re.sub(r'(?<![.\w])' + re.escape(name) + r'\b',
                            'raisimpy_rayrai.' + name, annotation)
    return annotation

for path in (output / 'raisimpy_rayrai').glob('*.pyi'):
    namespace = '' if path.stem == '__init__' else path.stem
    stack = []
    lines = []
    for line in path.read_text().splitlines():
        indent = len(line) - len(line.lstrip())
        if line.strip() and not line.lstrip().startswith(('@', '#')):
            while stack and indent <= stack[-1][0]:
                stack.pop()
        match = re.match(r'\s*class (\w+)', line)
        if match:
            stack.append((indent, match.group(1)))
        method = re.match(r'\s*def (\w+)\(', line)
        if method and ' -> object:' in line:
            owner = '.'.join(([namespace] if namespace else []) + [name for _, name in stack])
            candidates = returns.get((owner, method.group(1)))
            if candidates:
                annotation = ' | '.join(sorted(candidates))
                if namespace:
                    annotation = qualify_submodule_return(annotation)
                line = line.replace(' -> object:', ' -> ' + annotation + ':')
        lines.append(line)
    text = '\n'.join(lines) + '\n'
    ast.parse(text)
    path.write_text(text)

(output / 'raisimpy/rayrai.pyi').write_text('from raisimpy_rayrai import *\navailable: bool\n')
core = output / 'raisimpy/__init__.pyi'
core.write_text(core.read_text() + '\nfrom raisimpy_rayrai import RayraiWindow as RayraiWindow, RayraiViewer as RayraiViewer\n')

for stub in output.rglob("*.pyi"):
    ast.parse(stub.read_text(), filename=str(stub))
