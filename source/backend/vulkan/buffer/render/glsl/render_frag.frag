#version 450
precision highp float;
layout(location=0) in vec4 v0;
layout(location=1) in vec4 v1;
layout(location=2) in vec4 v2;

layout(location=0) out vec4 mask;
layout(location=1) out vec4 r0;
layout(location=2) out vec4 r1;
layout(location=3) out vec4 r2;

void main() {
    mask = vec4(1);
    r0 = v0;
    r1 = v1;
    r2 = v2;
}