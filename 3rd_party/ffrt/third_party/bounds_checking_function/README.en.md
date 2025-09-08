# libboundscheck

#### Description

- following the standard of C11 Annex K (bound-checking interfaces), functions of the common memory/string operation classes, such as memcpy_s, strcpy_s, are selected and implemented.

- other standard functions in C11 Annex K will be analyzed in the future and implemented in this organization if necessary.

- handles the release, update, and maintenance of bounds_checking_function.

#### Function List

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


#### Build

```
CC=gcc make
```
The generated Dynamic library libboundscheck.so is stored in the newly created directory lib.

#### How to use
1. Copy the libboundscheck.so to the library file directory, for example: "/usr/local/lib/".

2. To use the libboundscheck, add the “-lboundscheck” parameters to the compiler, for example: “gcc -g -o test test.c -lboundscheck”. 