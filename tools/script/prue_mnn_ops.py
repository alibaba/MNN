#!/usr/bin/env python3
import os
import sys

execute_treat_list = [
    "source/backend/cpu/CPUOPRegister.cpp",
    "source/backend/metal/MetalOPRegister.mm"
]


shape_treat_file = "source/shape/ShapeRegister.cpp"
import re

def extract_geometry_and_op_types(code: str):
    register_pattern = r'REGISTER_GEOMETRY\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*;'
    matches = re.findall(register_pattern, code)

    results = []

    for geometry_type, func_name in matches:
        func_pattern = rf'static\s+void\s+{re.escape(func_name)}\s*\(\s*\)\s*\{{(.*?)\}}'
        func_match = re.search(func_pattern, code, re.DOTALL)

        op_types = []
        if func_match:
            func_body = func_match.group(1)
            op_type_pattern = r'OpType_\w+'
            op_types = re.findall(op_type_pattern, func_body)

        results.append((geometry_type, op_types))

    return results

def load_op_set(op_file):
    ops = set()
    if os.path.exists(op_file):
        with open(op_file, 'r') as f:
            for line in f:
                op = line.strip()
                if op and not op.startswith('#'):
                    ops.add("OpType_" + op)
    return ops
if __name__ == '__main__':
    if len(sys.argv) <3:
        print("Usage: python3 prue_mnn_ops.py op.txt ${MNN_ROOT}")
        exit()
    ops = load_op_set(sys.argv[1])
    # push default op
    exes = ops
    exes.add("OpType_Cast")
    exes.add("OpType_FloatToInt8")
    exes.add("OpType_Int8ToFloat")
    exes.add("OpType_Int8ToFloat")
    exes.add("OpType_Raster")
    exes.add("OpType_While")

    mnnroot = sys.argv[2]
    for filename in execute_treat_list:
        lines = []
        with open(mnnroot + "/" + filename, 'r') as f:
            lines = f.read().split('\n')
        full = ""
        for line in lines:
            valid = line.find('OpType') < 0
            if not valid:
                for op in exes:
                    if line.find('_' + op + '_') >= 0:
                        valid = True
            if valid:
                full += line + '\n'
        with open(mnnroot + "/" + filename, 'w') as f:
            f.write(full)
    lines = []
    with open(mnnroot + "/" + shape_treat_file, 'r') as f:
        lines = f.read().split('\n')
    full = ""
    for line in lines:
        valid = line.find('OpType') < 0
        if not valid:
            for op in ops:
                if line.find('_' + op + '_') >= 0:
                    valid = True
        if valid:
            full += line + '\n'
    with open(mnnroot + "/" + shape_treat_file, 'w') as f:
        f.write(full)
    
    # Search Geometry File to make op type and file link
    geoDir = mnnroot + "/" + "source/geometry"
    fileNames = os.listdir(geoDir)
    funcNames = []
    geometry_treat_file = mnnroot + "/" + "source/geometry/GeometryOPRegister.cpp"
    need_delete_geo = []
    for fi in fileNames:
        if ".hpp" in fi:
            continue
        f = os.path.join(geoDir, fi)
        with open(f) as fr:
            res = extract_geometry_and_op_types(fr.read())
            for r, oplist in res:
                valid = False
                for op in oplist:
                    print(op)
                    if op in ops:
                        valid = True
                        break
                if not valid:
                    need_delete_geo.append(r)
    print("Remove Geometry Op:", need_delete_geo)
    lines = []
    with open(geometry_treat_file, 'r') as f:
        lines = f.read().split('\n')
    full = ""
    for line in lines:
        valid = True
        for op in need_delete_geo:
            if line.find('_' + op + '_') >=0:
                valid = False
        if valid:
            full += line + '\n'
    with open(geometry_treat_file, 'w') as f:
        f.write(full)


    
    

