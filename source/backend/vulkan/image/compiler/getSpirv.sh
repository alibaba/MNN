#!/bin/bash
glslangValidator -V $1 -Os -o __temp.spv
spirv-dis __temp.spv > $2

rm __temp.spv
