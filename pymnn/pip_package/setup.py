# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" setup tool """
from __future__ import print_function
import os

def get_version():
    root_dir = os.getenv('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.getcwd())))
    version_header = os.path.join(root_dir, 'include/MNN/MNNDefine.h')
    version_major = version_minor = version_patch = 'x'
    for line in open(version_header, 'rt').readlines():
        if '#define MNN_VERSION_MAJOR' in line:
            version_major = int(line.strip().split(' ')[-1])
        if '#define MNN_VERSION_MINOR' in line:
            version_minor = int(line.strip().split(' ')[-1])
        if '#define MNN_VERSION_PATCH' in line:
            version_patch = int(line.strip().split(' ')[-1])
    return '{}.{}.{}'.format(version_major, version_minor, version_patch)

import sys
import argparse
parser = argparse.ArgumentParser(description='build pymnn wheel')
parser.add_argument('--x86', dest='x86', action='store_true', default=False,
                    help='build wheel for 32bit arch, only usable on windows')
parser.add_argument('--version', dest='version', type=str, default=get_version(),
                    help='MNN dist version')
parser.add_argument('--serving', dest='serving', action='store_true', default=False,
                    help='build for internal serving, default False')
parser.add_argument('--deps', dest='deps', type=str, required=False,
                    help='MNN library deps')
parser.add_argument('--env', dest='env', type=str, required=False,
                    help='build environment, e.g. :daily/pre/production')
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

import platform
try:
   import numpy as np
except:
   print("import numpy failed")
from setuptools import setup, Extension, find_packages
from distutils import core
from distutils.core import Distribution
from distutils.errors import DistutilsArgError
IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')
BUILD_DIR = 'pymnn_build'
BUILD_TYPE = 'REL_WITH_DEB_INFO'
BUILD_ARCH = 'x64'
if args.x86:
    BUILD_ARCH = ''

def check_env_flag(name, default=''):
    """ check whether a env is set to Yes """
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']

def report(*args):
    """ print information """
    print(*args)

package_name = 'MNN'
USE_INTERNAL = False
USE_TRT      = False
USE_CUDA     = False
USE_OPENCL   = False
USE_VULKAN   = False
USE_RENDER   = False

if args.deps != None:
    if "trt" in args.deps:
        USE_TRT = True
    if "cuda" in args.deps:
        USE_CUDA = True
    if "opencl" in args.deps:
        USE_OPENCL = True
    if "vulkan" in args.deps:
        USE_VULKAN = True
    if "internal" in args.deps:
        USE_INTERNAL = True
    if "render" in args.deps:
        USE_RENDER = True

print ("USE_INTERNAL:", USE_INTERNAL)
print ("USE_TRT:", USE_TRT)
print ("USE_CUDA:", USE_CUDA)
print ("USE_OPENCL:", USE_OPENCL)
print ("USE_VULKAN:", USE_VULKAN)
print ("USE_RENDER:", USE_RENDER)

if os.path.isdir('../../schema/private'):
    package_name += '_Internal'
else:
    USE_INTERNAL = False

if USE_TRT:
    package_name += '_TRT'
if USE_CUDA:
    package_name += '_CUDA'
if USE_VULKAN:
    package_name += '_VULKAN'
if USE_OPENCL:
    package_name += '_OPENCL'
if USE_RENDER:
    package_name += '_RENDER'

print ('Building with python wheel with package name ', package_name)

version = args.version
depend_pip_packages = ['numpy']
README = os.path.join(os.getcwd(), "README.md")
with open(README) as f:
    long_description = f.read()

def configure_extension_build():
    r"""Configures extension build options according to system environment and user's choice.

    Returns:
       ext_modules, cmdclass, packages and entry_points as required in setuptools.setup.
    """
    ################################################################################
    # Configure compile flags
    ################################################################################


    if IS_WINDOWS:
        # /NODEFAULTLIB makes sure we only link to DLL runtime
        # and matches the flags set for protobuf and ONNX
        # extra_link_args = ['/NODEFAULTLIB:LIBCMT.LIB']
        # /MD links against DLL runtime
        # and matches the flags set for protobuf and ONNX
        # /Zi turns on symbolic debugging information in separate .pdb (which is same as MNN.pdb)
        # /EHa is about native C++ catch support for asynchronous
        # structured exception handling (SEH)
        # /DNOMINMAX removes builtin min/max functions
        # /wdXXXX disables warning no. XXXX
        # Some macro (related with __VA_ARGS__) defined in pymnn/src/util.h can not be process correctly
        # becase of MSVC bug, enable /experimental:preprocessor fix it (And Windows SDK >= 10.0.18362.1)
        extra_compile_args = ['/MT', '/Zi',
                              '/EHa', '/DNOMINMAX',
                              '/wd4267', '/wd4251', '/wd4522', '/wd4522', '/wd4838',
                              '/wd4305', '/wd4244', '/wd4190', '/wd4101', '/wd4996',
                              '/wd4275', '/experimental:preprocessor']
        extra_link_args = []
    else:
        extra_link_args = []
        extra_compile_args = [
            '-std=c++11',
            '-Wall',
            '-Wextra',
            '-Wno-strict-overflow',
            '-Wno-unused-parameter',
            '-Wno-missing-field-initializers',
            '-Wno-write-strings',
            '-Wno-unknown-pragmas',
            # This is required for Python 2 declarations that are deprecated in 3.
            '-Wno-deprecated-declarations',
            # Python 2.6 requires -fno-strict-aliasing, see
            # http://legacy.python.org/dev/peps/pep-3123/
            # We also depend on it in our code (even Python 3).
            '-fno-strict-aliasing',
            # Clang has an unfixed bug leading to spurious missing
            # braces warnings, see
            # https://bugs.llvm.org/show_bug.cgi?id=21629
            '-Wno-missing-braces',
        ]
        if check_env_flag('WERROR'):
            extra_compile_args.append('-Werror')
    extra_compile_args += ['-DPYMNN_EXPR_API', '-DPYMNN_NUMPY_USABLE', '-DPYMNN_OPENCV_API']
    if IS_LINUX and USE_INTERNAL:
        extra_compile_args += ['-DPYMNN_INTERNAL_SERVING']
        if args.env == 'daily':
            extra_compile_args += ['-DPYMNN_INTERNAL_SERVING_DAILY']
    root_dir = os.getenv('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.getcwd())))
    engine_compile_args = ['-DBUILD_OPTYPE', '-DPYMNN_TRAIN_API']
    engine_libraries = []
    engine_library_dirs = [os.path.join(root_dir, BUILD_DIR)]
    engine_library_dirs += [os.path.join(root_dir, BUILD_DIR, "tools", "train")]
    engine_library_dirs += [os.path.join(root_dir, BUILD_DIR, "tools", "cv")]
    engine_library_dirs += [os.path.join(root_dir, BUILD_DIR, "source", "backend", "tensorrt")]
    engine_library_dirs += [os.path.join(root_dir, BUILD_DIR, "source", "backend", "cuda")]
    if USE_TRT or USE_CUDA:
        # Note: TensorRT-5.1.5.0/lib should be set in $LIBRARY_PATH of the build system.
        engine_library_dirs += ['/usr/local/cuda/lib64/']

    # Logging is enabled on Linux. Add the dependencies.
    if IS_LINUX and USE_INTERNAL:
        engine_library_dirs += ['/usr/include/curl/']

    print(engine_library_dirs)
    engine_link_args = []
    engine_sources = [os.path.join(root_dir, "pymnn", "src", "MNN.cc")]
    if IS_LINUX and USE_INTERNAL:
        engine_sources += [os.path.join(root_dir, "pymnn", "src", "internal", "monitor_service.cc")]
        engine_sources += [os.path.join(root_dir, "pymnn", "src", "internal", "verify_service.cc")]
        engine_sources += [os.path.join(root_dir, "pymnn", "src", "internal", "http_util.cc")]
    engine_include_dirs = [os.path.join(root_dir, "include")]
    engine_include_dirs += [os.path.join(root_dir, "express")]
    engine_include_dirs += [os.path.join(root_dir, "express", "module")]
    engine_include_dirs += [os.path.join(root_dir, "source")]
    engine_include_dirs += [os.path.join(root_dir, "tools")]
    engine_include_dirs += [os.path.join(root_dir, "tools", "train", "source", "nn")]
    engine_include_dirs += [os.path.join(root_dir, "tools", "train", "source", "grad")]
    engine_include_dirs += [os.path.join(root_dir, "tools", "train", "source", "module")]
    engine_include_dirs += [os.path.join(root_dir, "tools", "train", "source", "parameters")]
    engine_include_dirs += [os.path.join(root_dir, "tools", "train", "source", "optimizer")]
    engine_include_dirs += [os.path.join(root_dir, "tools", "train", "source", "data")]
    engine_include_dirs += [os.path.join(root_dir, "tools", "train", "source", "transformer")]
    engine_include_dirs += [os.path.join(root_dir, "source", "core")]
    engine_include_dirs += [os.path.join(root_dir, "schema", "current")]
    engine_include_dirs += [os.path.join(root_dir, "3rd_party",\
                                          "flatbuffers", "include")]
    if IS_LINUX and USE_INTERNAL:
        engine_include_dirs += [os.path.join(root_dir, "3rd_party", "rapidjson")]
    # cv include
    engine_include_dirs += [os.path.join(root_dir, "tools", "cv", "include")]
    # llm include
    engine_include_dirs += [os.path.join(root_dir, "transformers", "llm", "engine", "include")]
    engine_include_dirs += [os.path.join(root_dir, "3rd_party")]
    engine_include_dirs += [np.get_include()]

    lib_files = []
    trt_depend = ['-lTRT_CUDA_PLUGIN', '-lnvinfer', '-lnvparsers', '-lnvinfer_plugin', '-lcudart']
    cuda_depend = ['-lMNN_Cuda_Main']
    engine_depend = ['-lMNN']

    # enable logging & model authentication on linux.
    if IS_LINUX and USE_INTERNAL:
        engine_depend += ['-lcurl', '-lssl', '-lcrypto']

    if USE_TRT:
        engine_depend += trt_depend
    if IS_DARWIN:
        lib_files += [('lib', [os.path.join(root_dir, BUILD_DIR, "libMNN.dylib")])]
        lib_files += [('lib', [os.path.join(root_dir, BUILD_DIR, "tools","converter", "libMNNConvertDeps.dylib")])]

    if USE_CUDA:
        engine_depend += cuda_depend
        lib_files += [('lib', [os.path.join(root_dir, BUILD_DIR, "source", "backend", "cuda", "libMNN_Cuda_Main.so")])]

    tools_compile_args = []
    tools_libraries = []
    tools_depend = ['-lMNN', '-lMNNConvertDeps', '-lprotobuf']
    tools_library_dirs = [os.path.join(root_dir, BUILD_DIR)]
    tools_library_dirs += [os.path.join(root_dir, BUILD_DIR, "tools", "converter")]
    tools_library_dirs += [os.path.join(root_dir, BUILD_DIR, "source", "backend", "tensorrt")]
    tools_library_dirs += [os.path.join(root_dir, BUILD_DIR, "source", "backend", "cuda")]
    tools_library_dirs += [os.path.join(root_dir, BUILD_DIR, "3rd_party", "protobuf", "cmake")]

    # add libTorch dependency
    torch_lib = None
    cmakecache = os.path.join(root_dir, BUILD_DIR, 'CMakeCache.txt')
    # llm
    for line in open(cmakecache, 'rt').readlines():
        if 'MNN_BUILD_LLM' in line:
            if 'ON' in line:
                extra_compile_args += ['-DPYMNN_LLM_API']
    # torch lib
    for line in open(cmakecache, 'rt').readlines():
        if 'TORCH_LIBRARY' in line:
            torch_lib = os.path.dirname(line[line.find('=')+1:])
            break
    if torch_lib is not None:
        tools_depend += ['-ltorch', '-ltorch_cpu', '-lc10']
        if IS_LINUX:
            tools_library_dirs += [torch_lib]
        elif IS_DARWIN:
            torch_path = os.path.dirname(torch_lib)
            tools_library_dirs += [torch_lib]
            lib_files += [('lib', [os.path.join(torch_lib, 'libtorch.dylib'), os.path.join(torch_lib, 'libtorch_cpu.dylib'),
                                  os.path.join(torch_lib, 'libc10.dylib')]),
                         ('.dylibs', [os.path.join(torch_lib, 'libiomp5.dylib')])]
            '''
            lib_files += [('lib', [os.path.join(torch_lib, 'libtorch.dylib'), os.path.join(torch_lib, 'libtorch_cpu.dylib'),
                                  os.path.join(torch_lib, 'libc10.dylib')]),
                         ('.dylibs', [os.path.join(torch_path, '.dylibs', 'libiomp5.dylib')])]
            '''
    if USE_TRT or USE_CUDA:
        # Note: TensorRT-5.1.5.0/lib should be set in $LIBRARY_PATH of the build system.
        tools_library_dirs += ['/usr/local/cuda/lib64/']

    if IS_LINUX and USE_INTERNAL:
        tools_library_dirs += ['/usr/include/curl/']

    tools_link_args = []
    tools_sources = [os.path.join(root_dir, "pymnn", "src", "MNNTools.cc")]
    tools_sources += [os.path.join(root_dir, "tools", "quantization",\
                                     "calibration.cpp")]
    tools_sources += [os.path.join(root_dir, "tools", "quantization",\
                                     "TensorStatistic.cpp")]
    tools_sources += [os.path.join(root_dir, "tools", "quantization",\
                                     "quantizeWeight.cpp")]
    tools_sources += [os.path.join(root_dir, "tools", "quantization", "Helper.cpp")]
    tools_include_dirs = []
    tools_include_dirs += [os.path.join(root_dir, "tools", "converter",\
                                       "include")]
    tools_include_dirs += [os.path.join(root_dir, "tools", "converter",\
                                       "source", "tflite", "schema")]
    tools_include_dirs += [os.path.join(root_dir, "tools", "converter", "source")]
    tools_include_dirs += [os.path.join(root_dir, BUILD_DIR, "tools", "converter")]
    tools_include_dirs += [os.path.join(root_dir, "include")]
    tools_include_dirs += [os.path.join(root_dir, "tools")]
    tools_include_dirs += [os.path.join(root_dir, "tools", "quantization")]
    tools_include_dirs += [os.path.join(root_dir, "3rd_party",\
                                          "flatbuffers", "include")]
    tools_include_dirs += [os.path.join(root_dir, "3rd_party")]
    tools_include_dirs += [os.path.join(root_dir, "3rd_party", "imageHelper")]
    tools_include_dirs += [os.path.join(root_dir, "source", "core")]
    tools_include_dirs += [os.path.join(root_dir, "schema", "current")]
    tools_include_dirs += [os.path.join(root_dir, "source")]
    tools_include_dirs += [np.get_include()]

    # enable logging and model authentication on linux.
    if IS_LINUX and USE_INTERNAL:
        tools_depend += ['-lcurl', '-lssl', '-lcrypto']

    if USE_TRT:
        tools_depend += trt_depend

    if USE_CUDA:
        tools_depend += cuda_depend

    if IS_DARWIN:
        engine_link_args += ['-stdlib=libc++']
        engine_link_args += engine_depend
    if IS_LINUX:
        engine_link_args += ['-Wl,--whole-archive']
        engine_link_args += engine_depend
        engine_link_args += ['-fopenmp']
        engine_link_args += ['-Wl,--no-whole-archive']
    if IS_WINDOWS:
        engine_link_args += ['/WHOLEARCHIVE:MNN.lib']
    if IS_DARWIN:
        tools_link_args += tools_depend
    if IS_LINUX:
        tools_link_args += ['-Wl,--whole-archive']
        tools_link_args += tools_depend
        tools_link_args += ['-fopenmp']
        tools_link_args += ['-Wl,--no-whole-archive']
        tools_link_args += ['-lz']
    if IS_WINDOWS:
        tools_link_args += ['/WHOLEARCHIVE:MNN.lib']
        tools_link_args += ['/WHOLEARCHIVE:MNNConvertDeps.lib']
        tools_link_args += ['libprotobuf.lib'] # use wholearchive will cause lnk1241 (version.rc specified)

    if BUILD_TYPE == 'DEBUG':
        # Need pythonxx_d.lib, which seem not exist in miniconda ?
        if IS_WINDOWS:
            extra_compile_args += ['/DEBUG', '/UNDEBUG', '/DDEBUG', '/Od', '/Ob0', '/MTd']
            extra_link_args += ['/DEBUG', '/UNDEBUG', '/DDEBUG', '/Od', '/Ob0', '/MTd']
        else:
            extra_compile_args += ['-O0', '-g']
            extra_link_args += ['-O0', '-g']

    if BUILD_TYPE == 'REL_WITH_DEB_INFO':
        if IS_WINDOWS:
            extra_compile_args += ['/DEBUG']
            extra_link_args += ['/DEBUG', '/OPT:REF', '/OPT:ICF']
        else:
            extra_compile_args += ['-g']
            extra_link_args += ['-g']

# compat with py39
    def make_relative_rpath(path):
        """ make rpath """
        if IS_DARWIN:
            # conda: dylibs install at site-packages/MNN_*/lib/
            # not conda: dylibs instal at .../lib/ for .../lib/python*/site-packages/_mnncengine.cpython-*-darwin.so
            return [f'-Wl,-rpath,@loader_path/../../../{path},-rpath,@loader_path/{path}']
        elif IS_WINDOWS:
            return []
        else:
            return ['-Wl,-rpath,$ORIGIN/' + path]

    ################################################################################
    # Declare extensions and package
    ################################################################################
    extensions = []
    packages = find_packages()
    engine = Extension("_mnncengine",\
                    libraries=engine_libraries,\
                    sources=engine_sources,\
                    language='c++',\
                    extra_compile_args=engine_compile_args + extra_compile_args,\
                    include_dirs=engine_include_dirs,\
                    library_dirs=engine_library_dirs,\
                    extra_link_args=engine_link_args + extra_link_args\
                        + make_relative_rpath('lib'))
    extensions.append(engine)
    tools = Extension("_tools",\
                    libraries=tools_libraries,\
                    sources=tools_sources,\
                    language='c++',\
                    extra_compile_args=tools_compile_args + extra_compile_args,\
                    include_dirs=tools_include_dirs,\
                    library_dirs=tools_library_dirs,\
                    extra_link_args=tools_link_args + extra_link_args\
                        + make_relative_rpath('lib'))
    extensions.append(tools)
    # These extensions are built by cmake and copied manually in build_extensions()
    # inside the build_ext implementaiton

    cmdclass = {}
    entry_points = {
        'console_scripts': [
            'mnnconvert = MNN.tools.mnnconvert:main',
            'mnnquant = MNN.tools.mnnquant:main',
            'mnn = MNN.tools.mnn:main'
        ]
    }

    return extensions, cmdclass, packages, entry_points, lib_files

# post run, warnings, printed at the end to make them more visible
build_update_message = """
    It is no longer necessary to use the 'build' or 'rebuild' targets

    To install:
      $ python setup.py install
    To develop locally:
      $ python setup.py develop
    To force cmake to re-generate native build files (off by default):
      $ python setup.py develop --cmake
"""

if __name__ == '__main__':
    # Parse the command line and check the arguments
    # before we proceed with building deps and setup
    dist = Distribution()
    dist.script_name = sys.argv[0]
    dist.script_args = sys.argv[1:]
    try:
        ok = dist.parse_command_line()
    except DistutilsArgError as msg:
        raise SystemExit(core.gen_usage(dist.script_name) + "\nerror: %s" % msg)
    if not ok:
        sys.exit()

    extensions, cmdclass, packages, entry_points, lib_files = configure_extension_build()

    setup(
        name=package_name,
        version=version,
        description=("C methods for MNN Package"),
        long_description=long_description,
        ext_modules=extensions,
        cmdclass=cmdclass,
        packages=packages,
        data_files=lib_files,
        entry_points=entry_points,
        install_requires=depend_pip_packages,
        url='https://www.yuque.com/mnn/en/usage_in_python',
        download_url='https://github.com/alibaba/MNN',
        author='alibaba MNN Team',
        author_email='lichuan.wlc@alibaba-inc.com',
        python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
        # PyPI package information.
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: C++',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='BSD-3',
        keywords='MNN Engine',
    )
