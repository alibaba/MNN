
from setuptools import setup, find_packages

import os
res = os.system("cd mnncompress/common;protoc --python_out=./ MNN_compression.proto")
assert res == 0, "protobuffer not generated, please make sure you have the protoc compiler installed."

import mnncompress


setup(
    name = "mnncompress",
    version = mnncompress.__version__,
    description = "model comprssion tools for MNN",
    author = "Yang Yafeng",
    author_email = "nickyoung.yyf@alibaba-inc.com",
    url = "https://github.com/alibaba/MNN",
    packages = find_packages(),
    zip_safe = False,
    platforms = "any",
    install_requires = [
        "aliyun-log-python-sdk",
        "tensorly==0.4.5"
    ],
)
