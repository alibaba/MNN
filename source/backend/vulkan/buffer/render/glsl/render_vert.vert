#version 450
precision highp float;
layout(location=0) in vec4 pos;
layout(location=1) in vec4 i0;
layout(location=2) in vec4 i1;
layout(location=3) in vec4 i2;

layout(location=0) out vec4 v0;
layout(location=1) out vec4 v1;
layout(location=2) out vec4 v2;

void main() {
    v0 = i0;
    v1 = i1;
    v2 = i2;
    gl_Position = pos;
}