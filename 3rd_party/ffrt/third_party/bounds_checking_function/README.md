# libboundscheck

#### 介绍
- 遵循C11 Annex K (Bounds-checking interfaces)的标准，选取并实现了常见的内存/字符串操作类的函数，如memcpy_s、strcpy_s等函数。
- 未来将分析C11 Annex K中的其他标准函数，如果有必要，将在该组织中实现。
- 处理边界检查函数的版本发布、更新以及维护。

#### 函数清单

- memcpy_s
- wmemcpy_s
- memmove_s
- wmemmove_s
- memset_s
- strcpy_s
- wcscpy_s
- strncpy_s
- wcsncpy_s
- strcat_s
- wcscat_s
- strncat_s
- wcsncat_s
- strtok_s
- wcstok_s
- sprintf_s
- swprintf_s
- vsprintf_s
- vswprintf_s
- snprintf_s
- vsnprintf_s
- scanf_s
- wscanf_s
- vscanf_s
- vwscanf_s
- fscanf_s
- fwscanf_s
- vfscanf_s
- vfwscanf_s
- sscanf_s
- swscanf_s
- vsscanf_s
- vswscanf_s
- gets_s

#### 构建方法

运行命令
```
make CC=gcc
```
生成的动态库libboundscheck.so存放在新创建的lib目录下。

#### 使用方法
1. 将构建生成的动态库libboundscheck.so放到库文件目录下，例如："/usr/local/lib/"。

2. 为使用libboundscheck，编译程序时需增加编译参数"-lboundscheck"，例如："gcc -g -o test test.c -lboundscheck"。