#!/usr/bin/env bash
# Prevent committing files larger than 1MB.
#
# Auto-generated source files (embedded shader hex/byte arrays, precompiled
# CUDA cubins, generated protobuf/flatbuffer code) are legitimately large and
# already tracked. Exclude them so regenerating them does not trip the limit.
# Keep in sync with the generated-file list in clang-format-diff.sh.
GENERATED_EXCLUDES=(
    # Vulkan embedded shaders
    ':(exclude)*/AllShader.h'
    ':(exclude)*/AllShader.cpp'
    ':(exclude)*/AllShaderRender.h'
    ':(exclude)*/AllShaderRender.cpp'
    # Metal embedded shaders
    ':(exclude)*/AllShader.hpp'
    ':(exclude)*/AllRenderShader.hpp'
    ':(exclude)*/AllRenderShader.cpp'
    # CUDA precompiled kernels
    ':(exclude)*.cubin.cpp'
    # Generated protobuf / flatbuffer code
    ':(exclude)*/generated/*.pb.h'
    ':(exclude)*/generated/*.pb.cc'
    ':(exclude)*/schema/*_generated.h'
)

MAX_SIZE=1048576
failed=0

git diff --cached --name-only --diff-filter=ACMR -z -- "${GENERATED_EXCLUDES[@]}" | while IFS= read -r -d '' f; do
    size=$(wc -c < "$f" 2>/dev/null || echo 0)
    if [ "$size" -gt "$MAX_SIZE" ]; then
        echo "ERROR: $f is $((size / 1024))KB, exceeds 1MB limit"
        # Write to temp file since subshell can't set parent variable
        echo 1 > /tmp/.pre-commit-large-file-failed
    fi
done

if [ -f /tmp/.pre-commit-large-file-failed ]; then
    rm -f /tmp/.pre-commit-large-file-failed
    exit 1
fi

exit 0
