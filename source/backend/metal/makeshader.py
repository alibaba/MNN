#!/usr/bin/python
import sys
import os
import re
import subprocess

gOutputHeadFile = "AllShader.hpp"
gOutputSourceFile = "AllShader.cpp"

def findAllShader(path):
    output = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".metal"):
                output.append(os.path.join(root, file))
    return output

def getName(fileName):
    s1 = fileName.replace("/", "_")
    s1 = s1.replace(".", "_")
    return s1

def generateFile(headfile, sourcefile, shaders):
    lasthead = headfile.split('/')
    lasthead = lasthead[len(lasthead)-1]
    # Only the non-render shaders are handled by packshader.py, so only they get
    # the MNN_METAL_PACK_SHADER gating that swaps the literals for blobs.
    packable = (lasthead == gOutputHeadFile)

    h = "#ifndef MNN_METAL_SHADER_AUTO_GENERATE_H\n#define MNN_METAL_SHADER_AUTO_GENERATE_H\n"
    cpp = "#include \"" + lasthead +"\"\n"
    if packable:
        h += "// With MNN_METAL_PACK_SHADER the shader text is compressed at build time and\n"
        h += "// these names become accessor macros supplied by the generated header.\n"
        h += "#ifdef MNN_METAL_PACK_SHADER\n"
        h += "#include \"MetalPackedShader.hpp\"\n"
        h += "#else\n"
        cpp += "// The literal definitions below are replaced by the compressed blobs in\n"
        cpp += "// MetalPackedShader.cpp when MNN_METAL_PACK_SHADER is enabled.\n"
        cpp += "#ifndef MNN_METAL_PACK_SHADER\n"
    mapcpp = "#include \"ShaderMap.hpp\"\n"
    mapcpp += '#include \"AllShader.hpp\"\n'
    mapcpp += 'namespace MNN {\n'
    mapcpp += 'void ShaderMap::init() {\n'
    for s in shaders:
        name = getName(s)
        print(name)
        h += "extern const char* " + name + ";\n";
        cpp += "const char* " + name + " = \n";
        spaceReg = re.compile(' +')
        with open(s) as f:
            lines = f.read().split("\n")
            for l in lines:
                if (len(l) < 1):
                    continue
                if l.find('#include') >= 0:
                    continue
                if l.find('#pragma clang') >= 0:
                    continue
                if l.find('\\') >= 0:
                    l = l.replace('\\', '')
                else:
                    l = l + "\\n"
                l = l.replace('\t', '')
                l = l.replace('ftype', 'M')
                l = l.replace('value', 'V')
                l = spaceReg.sub(' ', l)
                l = l.replace(', ', ',')
                l = l.replace(' = ', '=')
                l = l.replace(' + ', '+')
                l = l.replace(' - ', '-')
                l = l.replace(' * ', '*')
                l = l.replace(' / ', '/')
                l = l.replace(' < ', '<')
                l = l.replace(' > ', '>')
                cpp += "\""+l+"\"\n"
        cpp += ";\n"
        mapcpp += 'mMaps.insert(std::make_pair(\"' + name + '\", ' + name + "));\n"
    mapcpp += '}\n}\n'
    if packable:
        h += "#endif /* MNN_METAL_PACK_SHADER */\n"
        cpp += "#endif /* MNN_METAL_PACK_SHADER */\n"
    h+= "#endif"
    with open(headfile, "w") as f:
        f.write(h);
    with open(sourcefile, "w") as f:
        f.write(cpp);
    with open('ShaderMap.cpp', 'w') as f:
        f.write(mapcpp)

if __name__ == '__main__':
    renderPath = "render"
    if os.path.isdir(renderPath):
        shaders = findAllShader("render/shader")
        generateFile(os.path.join(renderPath, "AllRenderShader.hpp"), os.path.join(renderPath, "AllRenderShader.cpp"), shaders)
    gDefaultPath = "shader"
    shaders = findAllShader(gDefaultPath)
    generateFile(gOutputHeadFile, gOutputSourceFile, shaders);
