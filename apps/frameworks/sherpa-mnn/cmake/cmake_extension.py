# cmake/cmake_extension.py
# Copyright (c)  2023  Xiaomi Corporation
#
# flake8: noqa

import os
import platform
import shutil
import sys
from pathlib import Path

import setuptools
from setuptools.command.build_ext import build_ext


def is_for_pypi():
    ans = os.environ.get("SHERPA_ONNX_IS_FOR_PYPI", None)
    return ans is not None


def is_macos():
    return platform.system() == "Darwin"


def is_windows():
    return platform.system() == "Windows"


def is_linux():
    return platform.system() == "Linux"


def is_arm64():
    return platform.machine() in ["arm64", "aarch64"]


def is_x86():
    return platform.machine() in ["i386", "i686", "x86_64"]


def enable_alsa():
    build_alsa = os.environ.get("SHERPA_ONNX_ENABLE_ALSA", None)
    return build_alsa and is_linux() and (is_arm64() or is_x86())


def get_binaries():
    binaries = [
        "sherpa-onnx",
        "sherpa-onnx-keyword-spotter",
        "sherpa-onnx-microphone",
        "sherpa-onnx-microphone-offline",
        "sherpa-onnx-microphone-offline-audio-tagging",
        "sherpa-onnx-microphone-offline-speaker-identification",
        "sherpa-onnx-offline",
        "sherpa-onnx-offline-audio-tagging",
        "sherpa-onnx-offline-language-identification",
        "sherpa-onnx-offline-punctuation",
        "sherpa-onnx-offline-speaker-diarization",
        "sherpa-onnx-offline-tts",
        "sherpa-onnx-offline-tts-play",
        "sherpa-onnx-offline-websocket-server",
        "sherpa-onnx-online-punctuation",
        "sherpa-onnx-online-websocket-client",
        "sherpa-onnx-online-websocket-server",
        "sherpa-onnx-vad-microphone",
        "sherpa-onnx-vad-microphone-offline-asr",
        "sherpa-onnx-vad-with-offline-asr",
    ]

    if enable_alsa():
        binaries += [
            "sherpa-onnx-alsa",
            "sherpa-onnx-alsa-offline",
            "sherpa-onnx-alsa-offline-speaker-identification",
            "sherpa-onnx-offline-tts-play-alsa",
            "sherpa-onnx-vad-alsa",
            "sherpa-onnx-alsa-offline-audio-tagging",
        ]

    if is_windows():
        binaries += [
            "onnxruntime.dll",
            "sherpa-onnx-c-api.dll",
            "sherpa-onnx-cxx-api.dll",
        ]

    return binaries


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            # In this case, the generated wheel has a name in the form
            # sherpa-xxx-pyxx-none-any.whl
            if is_for_pypi() and not is_macos():
                self.root_is_pure = True
            else:
                # The generated wheel has a name ending with
                # -linux_x86_64.whl
                self.root_is_pure = False

except ImportError:
    bdist_wheel = None


def cmake_extension(name, *args, **kwargs) -> setuptools.Extension:
    kwargs["language"] = "c++"
    sources = []
    return setuptools.Extension(name, sources, *args, **kwargs)


class BuildExtension(build_ext):
    def build_extension(self, ext: setuptools.extension.Extension):
        # build/temp.linux-x86_64-3.8
        os.makedirs(self.build_temp, exist_ok=True)

        # build/lib.linux-x86_64-3.8
        os.makedirs(self.build_lib, exist_ok=True)

        out_bin_dir = Path(self.build_lib).parent / "sherpa_onnx" / "bin"
        install_dir = Path(self.build_lib).resolve() / "sherpa_onnx"

        sherpa_onnx_dir = Path(__file__).parent.parent.resolve()

        cmake_args = os.environ.get("SHERPA_ONNX_CMAKE_ARGS", "")
        make_args = os.environ.get("SHERPA_ONNX_MAKE_ARGS", "")
        system_make_args = os.environ.get("MAKEFLAGS", "")

        if cmake_args == "":
            cmake_args = "-DCMAKE_BUILD_TYPE=Release"

        extra_cmake_args = f" -DCMAKE_INSTALL_PREFIX={install_dir} "
        extra_cmake_args += " -DBUILD_SHARED_LIBS=ON "
        extra_cmake_args += " -DBUILD_PIPER_PHONMIZE_EXE=OFF "
        extra_cmake_args += " -DBUILD_PIPER_PHONMIZE_TESTS=OFF "
        extra_cmake_args += " -DBUILD_ESPEAK_NG_EXE=OFF "
        extra_cmake_args += " -DBUILD_ESPEAK_NG_TESTS=OFF "
        extra_cmake_args += " -DSHERPA_ONNX_ENABLE_C_API=ON "

        extra_cmake_args += " -DSHERPA_ONNX_BUILD_C_API_EXAMPLES=OFF "
        extra_cmake_args += " -DSHERPA_ONNX_ENABLE_CHECK=OFF "
        extra_cmake_args += " -DSHERPA_ONNX_ENABLE_PYTHON=ON "
        extra_cmake_args += " -DSHERPA_ONNX_ENABLE_PORTAUDIO=ON "
        extra_cmake_args += " -DSHERPA_ONNX_ENABLE_WEBSOCKET=ON "

        if "PYTHON_EXECUTABLE" not in cmake_args:
            print(f"Setting PYTHON_EXECUTABLE to {sys.executable}")
            cmake_args += f" -DPYTHON_EXECUTABLE={sys.executable}"

        cmake_args += extra_cmake_args

        if is_windows():
            build_cmd = f"""
         cmake {cmake_args} -B {self.build_temp} -S {sherpa_onnx_dir}
         cmake --build {self.build_temp} --target install --config Release -- -m:2
            """
            print(f"build command is:\n{build_cmd}")
            ret = os.system(
                f"cmake {cmake_args} -B {self.build_temp} -S {sherpa_onnx_dir}"
            )
            if ret != 0:
                raise Exception("Failed to configure sherpa")

            ret = os.system(
                f"cmake --build {self.build_temp} --target install --config Release -- -m:2"  # noqa
            )
            if ret != 0:
                raise Exception("Failed to build and install sherpa")
        else:
            if make_args == "" and system_make_args == "":
                print("for fast compilation, run:")
                print('export SHERPA_ONNX_MAKE_ARGS="-j"; python setup.py install')
                print('Setting make_args to "-j4"')
                make_args = "-j4"

            if "-G Ninja" in cmake_args:
                build_cmd = f"""
                    cd {self.build_temp}
                    cmake {cmake_args} {sherpa_onnx_dir}
                    ninja {make_args} install
                """
            else:
                build_cmd = f"""
                    cd {self.build_temp}

                    cmake {cmake_args} {sherpa_onnx_dir}

                    make {make_args} install/strip
                """
            print(f"build command is:\n{build_cmd}")

            ret = os.system(build_cmd)
            if ret != 0:
                raise Exception(
                    "\nBuild sherpa-onnx failed. Please check the error message.\n"
                    "You can ask for help by creating an issue on GitHub.\n"
                    "\nClick:\n\thttps://github.com/k2-fsa/sherpa-onnx/issues/new\n"  # noqa
                )

        suffix = ".exe" if is_windows() else ""
        # Remember to also change setup.py

        binaries = get_binaries()

        for f in binaries:
            suffix = "" if ".dll" in f else suffix
            src_file = install_dir / "bin" / (f + suffix)
            if not src_file.is_file():
                src_file = install_dir / "lib" / (f + suffix)
            if not src_file.is_file():
                src_file = install_dir / ".." / (f + suffix)

            print(f"Copying {src_file} to {out_bin_dir}/")
            shutil.copy(f"{src_file}", f"{out_bin_dir}/")

        shutil.rmtree(f"{install_dir}/bin")
        shutil.rmtree(f"{install_dir}/share")
        shutil.rmtree(f"{install_dir}/lib/pkgconfig")

        if is_macos():
            os.remove(f"{install_dir}/lib/libonnxruntime.dylib")

        if is_windows():
            shutil.rmtree(f"{install_dir}/lib")
