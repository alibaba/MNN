#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import subprocess
import re

ignore_dirs = ["3rdParty", "/build", "/Pods", "/schema/", "backend/OpenCL/CL", "backend/OpenCL/half"]
ignore_files = [
    "HalideRuntime.h", "cxxopts.hpp",
    "SkNx_neon.h", "SkNx.h", 
    "Matrix_CV.cpp", "Matrix.h",
    "vulkan_wrapper.cpp", "vulkan_wrapper.h", "vk_platform.h", "vulkan.h", "vulkan_android.h", "vulkan_core.h", 
    "CPUFixedPoint.hpp", "OptimizedComputer.hpp", "OptimizedComputer.cpp",
    "AllShader.h", "AllShader.cpp", "VulkanShaderMap.cpp"
    ]
all_exts = [".c", ".cpp", ".h", ".hpp", ".m", ".mm", ".s", ".metal", ".cuh", '.cu']

header_template = \
"//\n"                                                  + \
"//  %s\n"                                              + \
"//  %s\n"                                              + \
"//\n"                                                  + \
"//  Created by MNN on %s.\n"                         + \
"//  Copyright © 2018, Alibaba Group Holding Limited\n" + \
"//\n\n"

git_log_date_cmd = "git log --format=%%ai \"%s\" | tail -1 | cut -d' ' -f1 | tr - /"

wp = os.path.abspath(os.path.dirname(__file__))
g = os.walk(wp)  

def get_project(root):
    if "/test/" in root:
        return "MNNTests"
    elif "/tools/converter/" in root:
        return "MNNConverter"
    else: 
        return "MNN"

for root,dirs,files in g:
    if any(d in root for d in ignore_dirs):
        continue

    # ignore hidden files  
    files = [f for f in files if not f[0] == '.']
    dirs[:] = [d for d in dirs if not d[0] == '.']
    project = get_project(root)

    for file_name in files: 
        # filter
        if any(f in file_name for f in ignore_files):
            continue
        ext = os.path.splitext(file_name)[1]
        if ext == None or ext == "":
            continue
        ext = ext.lower()
        if not any(ext in s for s in all_exts):
            continue

        # get file added date
        file_path = os.path.join(root, file_name)
        cmd = git_log_date_cmd%(file_path)
        git_date = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read()[:-1]
        if len(git_date) != len("yyyy/MM/dd"):
            print("not tracing %s %s" % (file_path, git_date))
            continue

        # remove old license and empty line
        file_str = ""
        file_date = ""
        date_rex1 = re.compile(r".*(\d{4})/(\d\d?)/(\d\d?).*") # yyyy/MM/dd
        date_rex2 = re.compile(r".*(\d\d?)/(\d\d?)/(\d{4}).*") # dd/MM/yyyy

        with open(file_path, 'r') as f: 
            check = True
            for line in f:
                append = True
                if check:
                    striped = line.lstrip()
                    if len(striped) <= 0 or striped.startswith("//"):
                        append = False
                        match = date_rex1.match(line)
                        if match:
                            file_date = "%s/%s%s/%s%s" % (
                                match.group(1), 
                                "0" if len(match.group(2)) < 2 else "", match.group(2), 
                                "0" if len(match.group(3)) < 2 else "", match.group(3)
                            )
                        else:
                            match = date_rex2.match(line)
                            if match:
                                file_date = "%s/%s%s/%s%s" % (
                                    match.group(3), 
                                    "0" if len(match.group(2)) < 2 else "", match.group(2), 
                                    "0" if len(match.group(1)) < 2 else "", match.group(1)
                                )
                    else:
                        check = False
                if append:
                    file_str += line

        if len(file_date) != len("yyyy/MM/dd"):
            file_date = git_date

        # add new licence
        ins_licence = header_template%(file_name, project, file_date)
        file_str = ins_licence + file_str

        # replace
        print("replacing %s created at %s" % (file_path, file_date))
        with open(file_path, "w") as f:
            f.write(file_str)
