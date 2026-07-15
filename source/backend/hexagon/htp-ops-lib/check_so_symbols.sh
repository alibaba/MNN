#!/bin/bash

# 检查参数
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_so>"
    exit 1
fi

SO_PATH="$1"

# 检查文件是否存在
if [ ! -f "$SO_PATH" ]; then
    echo "Error: File $SO_PATH does not exist."
    exit 1
fi

# 获取 undefined symbols
NM_BIN="${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-nm"
if [ ! -x "$NM_BIN" ]; then
    echo "Error: hexagon-nm tool not found at $NM_BIN"
    exit 1
fi

# 运行 hexagon-nm 获取所有 undefined (U) 和 weak (w) 的符号
# 过滤掉系统库常见的合法未定义符号（如 __cxa_*, _Z*, __hexagon_*, qurt_*, HAP_*, std::*, libc 相关的等）
# 我们主要关注 htp_ops_* 或其他我们自己定义的但缺失实现的函数

echo "Checking for unexpected undefined symbols in $SO_PATH ..."

UNEXPECTED_SYMBOLS=$("$NM_BIN" -u "$SO_PATH" | awk '$1 == "U" {print $2}' | grep -vE "^_Z" | grep -vE "^__cxa" | grep -vE "^__hexagon" | grep -vE "^__gxx" | grep -vE "^__wrap" | grep -vE "^__extend" | grep -vE "^__trunc" | grep -vE "^_St" | grep -vE "^_WSt" | grep -vE "^_Btowc|_Get|_St|_Wc" | grep -vE "^__register|^__restore|^__save" | grep -vE "^qurt_" | grep -vE "^HAP_" | grep -vE "^dspqueue_" | grep -vE "^aligned_alloc|^abort|^ceilf|^close|^exp|^fabs|^fclose|^fflush|^fileno|^fopen|^fput|^fread|^fseek|^ftello|^fwrite|^getc|^getwc|^isatty|^isw|^mbr|^mbs|^mbt|^mem|^open|^posix_memalign|^puts|^read|^setbuf|^setlocale|^snprintf|^sqrtf|^sscanf|^str|^swprintf|^tanhf|^tow|^unget|^vasprintf|^vfprintf|^vsnprintf|^vsscanf|^wcr|^wcs|^wmem|^_Assert|^_Unwind_Resume")

if [ -n "$UNEXPECTED_SYMBOLS" ]; then
    echo -e "\033[31mERROR: Found unexpected undefined symbols in $SO_PATH:\033[0m"
    echo "$UNEXPECTED_SYMBOLS"
    echo -e "\033[31mThis will cause dlopen to fail on DSP and return code 44 (AEE_EFAILED) in FastRPC!\033[0m"
    exit 1
else
    echo -e "\033[32mSuccess: No unexpected undefined symbols found in $SO_PATH.\033[0m"
    exit 0
fi
