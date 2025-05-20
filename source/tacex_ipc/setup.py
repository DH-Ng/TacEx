"""Installation script for the 'tacex_ipc' python package.

Invoke with, for example:
# in .../source/tacex_ipc folder
pip install -e .

or with user arguments:
pip install -C--build-option=build_ext -C--build-option=--DCMAKE-CUDA-ARCHITECTURES=89 -C--build-option=--DUIPC-BUILD-PYBIND=1 .

-> custom build arguments don't work for editable install, due to build_temp method (see https://github.com/pypa/setuptools/issues/2491)
Workaround: use ENV variables (see https://github.com/pypa/setuptools/discussions/3969)
"""

import os
import toml

import re
import sys
import sysconfig
import platform
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

from pathlib import Path

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    # "cmake>=3.26",
    # "pybind11"
]

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):

    # hypens are automatically converted to underscores for attribute values
    # -> underscore directly isnt allowed for user options string
    user_options = build_ext.user_options + \
        [("DCMAKE-CUDA-ARCHITECTURES=", None, "Specify CUDA architecture,e.g. 89.")] + \
        [("DUIPC-BUILD-PYBIND=", "1", "Whether to build libuipc python bindings or not.")]    

    def initialize_options(self):
        super().initialize_options()
        self.DCMAKE_CUDA_ARCHITECTURES = None
        self.DUIPC_BUILD_PYBIND = "1"

    def finalize_options(self):
        # assert self.DCMAKE_CUDA_ARCHITECTURES in (None)
        super().finalize_options()

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        # use ENV variable as workaround
        self.DCMAKE_CUDA_ARCHITECTURES = os.environ.get('CMAKE_CUDA_ARCHITECTURES', None)
        print("cuda_architectures", self.DCMAKE_CUDA_ARCHITECTURES)

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DUIPC_BUILD_PYBIND=" + self.DUIPC_BUILD_PYBIND, # per default = 1
            "-DUIPC_DEV_MODE=1"
        ]
        if self.DCMAKE_CUDA_ARCHITECTURES is not None: # None means "use native cuda architecture"
            cmake_args += ["-DCMAKE_CUDA_ARCHITECTURES="+self.DCMAKE_CUDA_ARCHITECTURES]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = []#['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['-j8']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        self.build_dir = "build/" # where the compiled files are placed
        if not os.path.exists(self.build_dir):
            os.makedirs(self.build_dir)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_dir, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_dir)

# with (Path(__file__).resolve().parent / "README.md").open("r", encoding="utf-8") as f:
#     long_description = f.read()

# Installation operation
setup(
    name="tacex_ipc",
    packages=["tacex_ipc"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    ext_modules=[CMakeExtension("uipc", "libuipc")],
    cmdclass=dict(
        build_ext=CMakeBuild
    ),
    zip_safe=False,
)
