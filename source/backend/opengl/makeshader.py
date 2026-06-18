#!/usr/bin/python
import sys
import os
import subprocess

gDefaultPath = sys.argv[1]#"glsl"
gOutputHeadFile = sys.argv[2]#"AllShader.hpp"
gOutputSourceFile = sys.argv[3]#"AllShader.cpp"
def findAllShader(path):
    output = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".glsl"):
                output.append(os.path.join(root, file))
    return output

def getName(fileName):
    s1 = fileName.replace("/", "_")
    s1 = s1.replace(".", "_")
    return s1

def generateFile(headfile, sourcefile, shaders):
    h = "#ifndef OPENGL_GLSL_SHADER_AUTO_GENERATE_H\n#define OPENGL_GLSL_SHADER_AUTO_GENERATE_H\n"
    cpp = "#include \"AllShader.hpp\"\n"
    for s in shaders:
        name = getName(s)
        print(name)
        h += "extern const char* " + name + ";\n";
        cpp += "const char* " + name + " = \n";
        with open(s) as f:
            lines = f.read().split("\n")
            for l in lines:
                if (len(l) < 1):
                    continue
                cpp += "\""+l+"\\n\"\n"
        cpp += ";\n"
    h+= "#endif"
    with open(headfile, "w") as f:
        f.write(h);
    with open(sourcefile, "w") as f:
        f.write(cpp);

if __name__ == '__main__':
    shaders = findAllShader(gDefaultPath)
    generateFile(gOutputHeadFile, gOutputSourceFile, shaders);
