# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" setup tool """
from __future__ import print_function
import os
import sys
import platform
from setuptools import setup, Extension, find_packages
from distutils import core
from distutils.core import Distribution
from distutils.errors import DistutilsArgError
IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')
BUILD_DIR = 'pymnn_build'
BUILD_TYPE = 'RELEASE'
def check_env_flag(name, default=''):
    """ check whether a env is set to Yes """
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']

def report(*args):
    """ print information """
    print(*args)

package_name = os.getenv('MNN_PACKAGE_NAME', 'MNN')
version = '0.0.8'
depend_pip_packages = ['flatbuffers', 'pydot_ng', 'graphviz']
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
        # /Z7 turns on symbolic debugging information in .obj files
        # /EHa is about native C++ catch support for asynchronous
        # structured exception handling (SEH)
        # /DNOMINMAX removes builtin min/max functions
        # /wdXXXX disables warning no. XXXX
        extra_compile_args = ['/MT', '/Z7',
                              '/EHa', '/DNOMINMAX',
                              '/wd4267', '/wd4251', '/wd4522', '/wd4522', '/wd4838',
                              '/wd4305', '/wd4244', '/wd4190', '/wd4101', '/wd4996',
                              '/wd4275']
        if sys.version_info[0] == 2:
            report('Can not support MNN with Python 2.7 on Windows.')
            sys.exit(1)
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
    root_dir = os.getenv('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.getcwd())))
    engine_compile_args = []
    engine_libraries = []
    engine_library_dirs = [os.path.join(root_dir, BUILD_DIR)]
    engine_link_args = []
    engine_sources = [os.path.join(root_dir, "pymnn", "src", "MNN.cc")]
    engine_include_dirs = [os.path.join(root_dir, "include")]
    engine_depend = ['-lMNN']

    tools_compile_args = []
    tools_libraries = []
    tools_library_dirs = [os.path.join(root_dir, BUILD_DIR)]
    tools_library_dirs += [os.path.join(root_dir, BUILD_DIR, "tools", "converter")]
    tools_link_args = []
    tools_sources = [os.path.join(root_dir, "pymnn", "src", "MNNTools.cc")]
    tools_sources += [os.path.join(root_dir, "tools", "quantization",\
                                     "calibration.cpp")]
    tools_sources += [os.path.join(root_dir, "tools", "quantization",\
                                     "TensorStatistic.cpp")]
    tools_sources += [os.path.join(root_dir, "tools", "quantization",\
                                     "quantizeWeight.cpp")]
    tools_sources += [os.path.join(root_dir, "tools", "quantization", "Helper.cpp")]
    tools_include_dirs = [os.path.join(root_dir, "tools", "converter",\
                                       "source", "IR")]
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
    tools_depend = ['-lMNN', '-lMNNConvertDeps']
    engine_extra_link_args = []
    tools_extra_link_args = []
    if IS_DARWIN:
        engine_extra_link_args += ['-Wl,-all_load']
        engine_extra_link_args += engine_depend
        engine_extra_link_args += ['-Wl,-noall_load']
    if IS_LINUX:
        engine_extra_link_args += ['-Wl,--whole-archive']
        engine_extra_link_args += engine_depend
        engine_extra_link_args += ['-Wl,--no-whole-archive']
    if IS_WINDOWS:
        engine_extra_link_args += ['/WHOLEARCHIVE:MNN.lib']
    if IS_DARWIN:
        tools_extra_link_args += ['-Wl,-all_load']
        tools_extra_link_args += tools_depend
        tools_extra_link_args += ['/usr/local/lib/libprotobuf.a']
        tools_extra_link_args += ['-Wl,-noall_load']
    if IS_LINUX:
        tools_extra_link_args += ['-Wl,--whole-archive']
        tools_extra_link_args += tools_depend
        tools_extra_link_args += ['-l:libprotobuf.a']
        tools_extra_link_args += ['-Wl,--no-whole-archive']
        tools_extra_link_args += ['-lz']
    if IS_WINDOWS:
        tools_extra_link_args += ['/WHOLEARCHIVE:MNN.lib']
        tools_extra_link_args += ['/WHOLEARCHIVE:COMMON_LIB.lib']
        tools_extra_link_args += ['/WHOLEARCHIVE:tflite.lib']
        tools_extra_link_args += ['/WHOLEARCHIVE:onnx.lib']
        tools_extra_link_args += ['/WHOLEARCHIVE:optimizer.lib']
        tools_extra_link_args += ['/WHOLEARCHIVE:mnn_bizcode.lib']
        tools_extra_link_args += ['/WHOLEARCHIVE:caffe.lib']
        tools_extra_link_args += ['/WHOLEARCHIVE:tensorflow.lib']
        tools_extra_link_args += ['C:\\Users\\tianhang.yth\\Desktop\\protobuf\\vsprojects\\Release\\libprotobuf.lib']

    if BUILD_TYPE == 'DEBUG':
        if IS_WINDOWS:
            extra_link_args.append('/DEBUG:FULL')
        else:
            extra_compile_args += ['-O0', '-g']
            extra_link_args += ['-O0', '-g']

    if BUILD_TYPE == 'REL_WITH_DEB_INFO':
        if IS_WINDOWS:
            extra_link_args.append('/DEBUG:FULL')
        else:
            extra_compile_args += ['-g']
            extra_link_args += ['-g']


    def make_relative_rpath(path):
        """ make rpath """
        if IS_DARWIN:
            return '-Wl,-rpath,@loader_path/' + path
        elif IS_WINDOWS:
            return ''
        else:
            return '-Wl,-rpath,$ORIGIN/' + path

    ################################################################################
    # Declare extensions and package
    ################################################################################
    extensions = []
    packages = find_packages()
    MNN = Extension("MNN",\
                    libraries=engine_libraries,\
                    sources=engine_sources,\
                    language='c++',\
                    extra_compile_args=engine_compile_args + extra_compile_args,\
                    include_dirs=engine_include_dirs,\
                    library_dirs=engine_library_dirs,\
                    extra_link_args=engine_extra_link_args + engine_link_args\
                        + [make_relative_rpath('lib')])
    extensions.append(MNN)
    Tools = Extension("Tools",\
                    libraries=tools_libraries,\
                    sources=tools_sources,\
                    language='c++',\
                    extra_compile_args=tools_compile_args + extra_compile_args,\
                    include_dirs=tools_include_dirs,\
                    library_dirs=tools_library_dirs,\
                    extra_link_args=tools_extra_link_args +tools_link_args\
                        + [make_relative_rpath('lib')])
    extensions.append(Tools)
    # These extensions are built by cmake and copied manually in build_extensions()
    # inside the build_ext implementaiton

    cmdclass = {}
    entry_points = {
        'console_scripts': [
            'mnnconvert = MNNTools.mnnconvert:main',
            'mnnquant = MNNTools.mnnquant:main',
            'mnnvisual = MNNTools.mnnvisual:main',
            'mnnops = MNNTools.mnnops:main',
            'mnn = MNNTools.mnn:main'
        ]
    }

    return extensions, cmdclass, packages, entry_points

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

    extensions, cmdclass, packages, entry_points = configure_extension_build()

    setup(
        name=package_name,
        version=version,
        description=("C methods for MNN Package"),
        long_description=long_description,
        ext_modules=extensions,
        cmdclass=cmdclass,
        packages=packages,
        entry_points=entry_points,
        install_requires=depend_pip_packages,
        url='https://www.yuque.com/mnn/en/usage_in_python',
        download_url='https://github.com/MNN',
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
