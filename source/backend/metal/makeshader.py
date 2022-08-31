#!/usr/bin/python
import sys
import os
import re
gOutputHeadFile = "AllShader.hpp"
gOutputSourceFile = "AllShader.cpp"
def findAllShader(path):
    cmd = "find " + path + " -name \"*.metal\""
    vexs = os.popen(cmd).read().split('\n')
    output = []
    for f in vexs:
        if len(f)>1:
            output.append(f)
    return output

def getName(fileName):
    s1 = fileName.replace("/", "_")
    s1 = s1.replace(".", "_")
    return s1

def generateFile(headfile, sourcefile, shaders):
    h = "#ifndef MNN_METAL_SHADER_AUTO_GENERATE_H\n#define MNN_METAL_SHADER_AUTO_GENERATE_H\n"
    cpp = "#include \"AllShader.hpp\"\n"
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
    h+= "#endif"
    with open(headfile, "w") as f:
        f.write(h);
    with open(sourcefile, "w") as f:
        f.write(cpp);
    with open('ShaderMap.cpp', 'w') as f:
        f.write(mapcpp)

if __name__ == '__main__':
    gDefaultPath = sys.argv[1]#"glsl"
    shaders = findAllShader(gDefaultPath)
    generateFile(gOutputHeadFile, gOutputSourceFile, shaders);
