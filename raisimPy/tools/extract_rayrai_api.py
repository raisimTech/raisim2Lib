"""Extract public C++ declarations with libclang (maintainer tool, not needed to build)."""
import ctypes as C, json, pathlib, re, argparse, ctypes.util, tempfile, hashlib
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--libclang', default=ctypes.util.find_library('clang-23'))
parser.add_argument('--gcc-install-dir', default='/usr/lib/gcc/x86_64-linux-gnu/15')
options = parser.parse_args()
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'raisimPy/rayrai/api.json'
unit = tempfile.NamedTemporaryFile(mode='w', suffix='.cpp')
unit.write('#include "rayrai/RayraiWindow.hpp"\n#include "rayrai/MeshPreprocessCache.hpp"\n')
unit.flush()
lib=C.CDLL(options.libclang)
class Cursor(C.Structure):
    _fields_=[('kind',C.c_uint),('xdata',C.c_int),('data',C.c_void_p*3)]
class String(C.Structure):
    _fields_=[('data',C.c_void_p),('flags',C.c_uint)]
class Type(C.Structure):
    _fields_=[('kind',C.c_uint),('data',C.c_void_p*2)]
class Location(C.Structure):
    _fields_=[('ptr_data',C.c_void_p*2),('int_data',C.c_uint)]
class Range(C.Structure):
    _fields_=[('ptr_data',C.c_void_p*2),('begin_int_data',C.c_uint),('end_int_data',C.c_uint)]
def fn(name, result, *args):
    f=getattr(lib,name);f.restype=result;f.argtypes=list(args);return f
getstr=fn('clang_getCString',C.c_char_p,String)
dispose=fn('clang_disposeString',None,String)
def string(value):
    r=getstr(value);r=r.decode() if r else '';dispose(value);return r
kindname=fn('clang_getCursorKindSpelling',String,C.c_uint)
name=fn('clang_getCursorSpelling',String,Cursor)
curtype=fn('clang_getCursorType',Type,Cursor)
typename=fn('clang_getTypeSpelling',String,Type)
resulttype=fn('clang_getCursorResultType',Type,Cursor)
canonical=fn('clang_getCanonicalType',Type,Type)
parent=fn('clang_getCursorSemanticParent',Cursor,Cursor)
access=fn('clang_getCXXAccessSpecifier',C.c_uint,Cursor)
loc=fn('clang_getCursorLocation',Location,Cursor)
spellloc=fn('clang_getSpellingLocation',None,Location,C.POINTER(C.c_void_p),C.POINTER(C.c_uint),C.POINTER(C.c_uint),C.POINTER(C.c_uint))
filename=fn('clang_getFileName',String,C.c_void_p)
extent=fn('clang_getCursorExtent',Range,Cursor)
rangestart=fn('clang_getRangeStart',Location,Range)
rangeend=fn('clang_getRangeEnd',Location,Range)
numargs=fn('clang_Cursor_getNumArguments',C.c_int,Cursor)
getarg=fn('clang_Cursor_getArgument',Cursor,Cursor,C.c_uint)
isstatic=fn('clang_CXXMethod_isStatic',C.c_uint,Cursor)
isconst=fn('clang_CXXMethod_isConst',C.c_uint,Cursor)
isdefaultctor=fn('clang_CXXConstructor_isDefaultConstructor',C.c_uint,Cursor)
isdeleted=fn('clang_CXXMethod_isDeleted',C.c_uint,Cursor)
comment=fn('clang_Cursor_getRawCommentText',String,Cursor)
def point(location):
    f=C.c_void_p();line=C.c_uint();col=C.c_uint();off=C.c_uint()
    spellloc(location,C.byref(f),C.byref(line),C.byref(col),C.byref(off))
    return string(filename(f)) if f else '',line.value,off.value
files={}
def source(c):
    r=extent(c);file,_,a=point(rangestart(r));_,_,b=point(rangeend(r))
    if not file:return ''
    if file not in files:files[file]=pathlib.Path(file).read_bytes()
    return files[file][a:b].decode(errors='replace')
def qualified(c):
    names=[]
    while c.kind not in (0, 70, 71, 72, 73, 350):
        n=string(name(c))
        if n:names.append(n)
        c=parent(c)
    return '::'.join(reversed(names))
callback=C.CFUNCTYPE(C.c_uint,Cursor,Cursor,C.c_void_p)
visit=fn('clang_visitChildren',C.c_uint,Cursor,callback,C.c_void_p)
def children(c):
    out=[]
    @callback
    def cb(child,p,data):out.append(child);return 1
    visit(c,cb,None);return out
idx=fn('clang_createIndex',C.c_void_p,C.c_int,C.c_int)(0,0)
args=['-x','c++','--gcc-install-dir=' + options.gcc_install_dir,'-std=c++20','-DGLM_ENABLE_EXPERIMENTAL','-DSYSTEM_LINUX','-I' + str(ROOT / 'rayrai/include'),'-I' + str(ROOT / 'raisim/include'),'-I/usr/include/eigen3','-I/usr/include/SDL2']
argarray=(C.c_char_p*len(args))(*[a.encode() for a in args])
parse=fn('clang_parseTranslationUnit',C.c_void_p,C.c_void_p,C.c_char_p,C.POINTER(C.c_char_p),C.c_int,C.c_void_p,C.c_uint,C.c_uint)
tu=parse(idx,unit.name.encode(),argarray,len(args),None,0,64)
if not tu:raise RuntimeError('Parse failed')
diagcount=fn('clang_getNumDiagnostics',C.c_uint,C.c_void_p)(tu)
getdiag=fn('clang_getDiagnostic',C.c_void_p,C.c_void_p,C.c_uint)
diagstr=fn('clang_getDiagnosticSpelling',String,C.c_void_p)
for i in range(diagcount):print('diagnostic:',string(diagstr(getdiag(tu,i))))
root=fn('clang_getTranslationUnitCursor',Cursor,C.c_void_p)(tu)
records=[];seen=set()
def walk(c):
    k=string(kindname(c.kind));file,line,_=point(loc(c));q=qualified(c)
    if k in ('Namespace','TranslationUnit'):
        for ch in children(c):walk(ch)
    elif file.startswith(str(ROOT / 'rayrai/include/rayrai/')) and ('/shaders/' not in file or file.endswith('/shaders/Shader.hpp')):
        if k in ('ClassDecl','StructDecl','EnumDecl') and q and q not in seen:
            items=children(c)
            if not items:return
            seen.add(q)
            rec={'name':q,'kind':k,'file':file,'line':line,'public':access(c)!=3,'members':[]}
            for ch in items:
                kind=string(kindname(ch.kind));n=string(name(ch));a=access(ch)
                if kind in ('ClassDecl','StructDecl','EnumDecl'):walk(ch)
                if a not in (0,1):continue
                item={'name':n,'kind':kind,'type':string(typename(curtype(ch))),'canonical':string(typename(canonical(curtype(ch)))),'source':source(ch),'comment':string(comment(ch))}
                if kind in ('CXXMethod','CXXConstructor','FunctionDecl','CXXConversion'):
                    item['result']=string(typename(resulttype(ch)));item['resultcanonical']=string(typename(canonical(resulttype(ch))));item['static']=bool(isstatic(ch));item['const']=bool(isconst(ch));item['deleted']=bool(isdeleted(ch));item['default_ctor']=bool(isdefaultctor(ch)) if kind=='CXXConstructor' else False
                    item['args']=[]
                    for i in range(max(0,numargs(ch))):
                        ar=getarg(ch,i);t=curtype(ar)
                        item['args'].append({'name':string(name(ar)),'type':string(typename(t)),'canonical':string(typename(canonical(t))),'source':source(ar)})
                rec['members'].append(item)
            records.append(rec)
        elif k in ('FunctionDecl','FunctionTemplate') and q.startswith('raisin::'):
            records.append({'name':q,'kind':k,'file':file,'line':line,'source':source(c),'type':string(typename(curtype(c))),'resultcanonical':string(typename(canonical(resulttype(c)))),'args':[{'name':string(name(getarg(c,i))),'type':string(typename(curtype(getarg(c,i)))),'canonical':string(typename(canonical(curtype(getarg(c,i))))),'source':source(getarg(c,i))} for i in range(max(0,numargs(c)))]})
for c in children(root):walk(c)
for record in records:
    record['file'] = str(pathlib.Path(record['file']).relative_to(ROOT))
OUTPUT.write_text(json.dumps(records, indent=2) + '\n')
headers = sorted({r['file'] for r in records})
(OUTPUT.parent/'header_hashes.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in headers},indent=2)+'\n')
from collections import Counter
print('records',len(records),Counter(r['kind'] for r in records))
for rec in records:
    if rec['name'] in ('raisin::RayraiWindow','raisin::Visuals','raisin::Camera','raisin::InstancedVisuals'):
        print(rec['name'],Counter(m['kind'] for m in rec['members']))
