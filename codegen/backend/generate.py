import os
import sys
import argparse
from os.path import dirname
import json
import shutil
from typing import List, Dict
import datetime

MNN_ROOT = dirname(dirname(dirname(__file__))) # 3 level up
CODEGEN_PATH = os.path.join(MNN_ROOT, "codegen")
CODEGEN_BACKEND_PATH = os.path.join(CODEGEN_PATH, "backend")
BACKEND_PATH = os.path.join(MNN_ROOT, "source", "backend")
print("MNN_ROOT: " + MNN_ROOT)

def parse_str(s, kv):
    k, v = kv
    return s.replace("{"+k+"}", v)

def get_copyright() -> str:
    c = open(os.path.join(CODEGEN_BACKEND_PATH, "templates", "copyright.txt"), "rt").read()
    return c.replace("{CURRENT_TIME}", datetime.datetime.today().strftime('%Y/%m/%d'))

def parse_copyright(s: str, c: str):
    return s.replace("{COPYRIGHT}", c)

def get_extra_includes(xpu_name):
    include_path = os.path.join(CODEGEN_PATH, xpu_name, "core", "include.json")
    return json.decoder.JSONDecoder().decode(open(include_path, "rt").read())

def parse_includes(s: str, d: Dict[str, str], k: str, domain: str):
    extra_includes = ""
    for n in d[domain][k]:
        extra_includes += "#include \"{}\"\n".format(n)
    return s.replace("{EXTRA_INCLUDE_FILES}", extra_includes)

def get_runtime_params(xpu_name):
    include_path = os.path.join(CODEGEN_PATH, xpu_name, "core", "runtime_params.json")
    return json.decoder.JSONDecoder().decode(open(include_path, "rt").read())

def parse_RUNTIME_PARAMS(s: str, d: Dict[str, str]):
    params = ""
    for t, n in d.items():
        params += "{t} {n};\n\t".format(t=t, n=n)
    return s.replace("{RUNTIME_PARAMS}", params)

def genCore(xpu_name):
    template_path = os.path.join(CODEGEN_BACKEND_PATH, "templates", "core")
    XPUBackend_hpp = open(os.path.join(template_path, "{XPU}Backend.hpp"), "rt").read()
    XPUBackend_cpp = open(os.path.join(template_path, "{XPU}Backend.cpp"), "rt").read()
    parse_dict = json.decoder.JSONDecoder().decode(
                    open(os.path.join(CODEGEN_PATH, xpu_name, "core", "symbol.json"), "rt").read())
    # parse symbols
    for kv in parse_dict.items():
        XPUBackend_hpp = parse_str(XPUBackend_hpp, kv)
        XPUBackend_cpp = parse_str(XPUBackend_cpp, kv)
    XPUBackend_hpp_name = "{XPU}Backend.hpp".format(XPU=parse_dict["XPU"])
    XPUBackend_cpp_name = "{XPU}Backend.cpp".format(XPU=parse_dict["XPU"])
    # parse copyright
    copyright = get_copyright()
    XPUBackend_hpp = parse_copyright(XPUBackend_hpp, copyright.format(THIS_FILE_NAME=XPUBackend_hpp_name))
    XPUBackend_cpp = parse_copyright(XPUBackend_cpp, copyright.format(THIS_FILE_NAME=XPUBackend_cpp_name))
    # parse include files
    extra_includes = get_extra_includes(xpu_name)
    XPUBackend_hpp = parse_includes(XPUBackend_hpp, extra_includes, XPUBackend_hpp_name, "core")
    XPUBackend_cpp = parse_includes(XPUBackend_cpp, extra_includes, XPUBackend_cpp_name, "core")
    # parse Rumtime Params
    runtime_params = get_runtime_params(xpu_name)
    XPUBackend_hpp = parse_RUNTIME_PARAMS(XPUBackend_hpp, runtime_params)
    XPUBackend_cpp = parse_RUNTIME_PARAMS(XPUBackend_cpp, runtime_params)
    print(XPUBackend_hpp, file=open(os.path.join(BACKEND_PATH, xpu_name, "core", XPUBackend_hpp_name), "wt"))
    print(XPUBackend_cpp, file=open(os.path.join(BACKEND_PATH, xpu_name, "core", XPUBackend_cpp_name), "wt"))
    shutil.copy(src=os.path.join(CODEGEN_PATH, xpu_name, "core", "{XPU}BackendUtil.cpp".format(XPU=parse_dict["XPU"])),
                dst=os.path.join(BACKEND_PATH, xpu_name, "core", "{XPU}BackendUtil.cpp".format(XPU=parse_dict["XPU"])))


def genBackend(xpu_name):
    os.makedirs(os.path.join(BACKEND_PATH, xpu_name), exist_ok=True)
    os.makedirs(os.path.join(BACKEND_PATH, xpu_name, "core"), exist_ok=True)
    os.makedirs(os.path.join(BACKEND_PATH, xpu_name, "execution"), exist_ok=True)
    
    genCore(xpu_name)

    # genCMakeLists

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    xpu_name = args.name
    if xpu_name == None:
        print("Format: python generate.py --name [xpu-name]")
        exit(1)

    genBackend(xpu_name)