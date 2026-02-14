import os
import sys
import subprocess
import platform
import sysconfig

from setuptools import Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.core import setup

__CMAKE_PREFIX_PATH__ = None
__DEBUG__ = False
__WITH_LIBTORCH__ = False
__LIBTORCH_ROOT__ = None

if "--CMAKE_PREFIX_PATH" in sys.argv:
    index = sys.argv.index('--CMAKE_PREFIX_PATH')
    __CMAKE_PREFIX_PATH__ = sys.argv[index+1]
    sys.argv.remove("--CMAKE_PREFIX_PATH")
    sys.argv.remove(__CMAKE_PREFIX_PATH__)

if "--Debug" in sys.argv:
    index = sys.argv.index('--Debug')
    sys.argv.remove("--Debug")
    __DEBUG__ = True

if "--Libtorch" in sys.argv:
    sys.argv.remove("--Libtorch")
    __WITH_LIBTORCH__ = True
    script_dir = os.path.dirname(os.path.abspath(__file__))
    __LIBTORCH_ROOT__ = os.path.join(script_dir, "thirdParty", "libtorch")

    torch_dir_env = os.environ.get("Torch_DIR")
    cmake_prefix_env = os.environ.get("CMAKE_PREFIX_PATH")
    torch_config = os.path.join(__LIBTORCH_ROOT__, "share", "cmake", "Torch", "TorchConfig.cmake")
    needs_libtorch = not torch_dir_env and not cmake_prefix_env and not os.path.exists(torch_config)

    if needs_libtorch and not os.environ.get("LIBTORCH_SKIP_DOWNLOAD"):
        downloader = os.path.join(script_dir, "scripts", "get_libtorch.py")
        cmd = [sys.executable, downloader, "--dest", os.path.join(script_dir, "thirdParty")]
        url_override = os.environ.get("LIBTORCH_URL")
        if url_override:
            cmd += ["--url", url_override]
        else:
            channel = os.environ.get("LIBTORCH_CHANNEL", "nightly")
            cuda_tag = os.environ.get("LIBTORCH_CUDA", "cpu")
            cmd += ["--channel", channel, "--cuda", cuda_tag]
            version = os.environ.get("LIBTORCH_VERSION")
            if version:
                cmd += ["--version", version]
        subprocess.check_call(cmd)

    if __CMAKE_PREFIX_PATH__ is None and os.path.exists(torch_config):
        __CMAKE_PREFIX_PATH__ = __LIBTORCH_ROOT__

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        python_exec = sys.executable
        python_include = sysconfig.get_paths().get("include")
        python_prefix = sysconfig.get_config_var("prefix") or sys.prefix

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + python_exec,
                      '-DPython_EXECUTABLE=' + python_exec,
                      '-DPython_ROOT_DIR=' + python_prefix]
        if python_include:
            cmake_args.append('-DPython_INCLUDE_DIR=' + python_include)
            cmake_args.append('-DPython_INCLUDE_DIRS=' + python_include)

        if __CMAKE_PREFIX_PATH__ is not None:
            cmake_args.append('-DCMAKE_PREFIX_PATH=' + __CMAKE_PREFIX_PATH__)
        if __WITH_LIBTORCH__:
            cmake_args.append('-DRAISIMGYM_TORCH_WITH_LIBTORCH=ON')

        cfg = 'Debug' if __DEBUG__ else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        include_flag = f"-I{python_include}" if python_include else ""
        cxxflags = env.get('CXXFLAGS', '')
        env['CXXFLAGS'] = f"{cxxflags} {include_flag} -DVERSION_INFO=\\\"{self.distribution.get_version()}\\\"".strip()
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='raisim_gym_torch',
    version='2.0.0',
    author='Jemin Hwangbo',
    license="proprietary",
    packages=find_packages(),
    author_email='jemin.hwangbo@gmail.com',
    description='gym for raisim using torch.',
    long_description='',
    ext_modules=[CMakeExtension('_raisim_gym')],
    cmdclass=dict(build_ext=CMakeBuild),
    include_package_data=True,
    zip_safe=False,
)
