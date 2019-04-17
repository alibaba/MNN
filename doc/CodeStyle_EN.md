[中文版本](CodeStyle_CN.md)

# Format Tools

We use `clang-format` and `git-clang-format` to unify C, C++, and Objective-C code styles. The code style is described using `.clang-format` file in the project root directory.

When adding files, use clang-format to adjust the code style:

``` shell
clang-format -i /path/to/new_file
```

When modifying a file, use git-clang-format to adjust the code style of the new section:

``` shell
cd /path/to/MNN
git-clang-format
```


# Code Style

For C, C++, and Objective-C code, we choose the Google code style. However, the following options were adjusted:

Project                                                                        | Modification
-------------------------------------------------------------------------------|-----------------------------
AccessModifierOffset (access permission correction offset)                     | from -1 to -4
AlignConsecutiveAssignments (Continuous assignment alignment)                  | Changed from false to true
ColumnLimit (column width limit)                                               | from 80 to 120
IndentWidth (Indent Width)                                                     | Change from 2 to 4
ObjCBlockIndentWidth (ObjC Block indent width )                                | from 2 to 4
ObjCSpaceAfterProperty (Observed after ObjCSpaceAfterProperty ObjC property)   | changed from false to true
SpacesBeforeTrailingComments (Number of spaces before the end of the comment)  | Changed from 2 to 1

## Naming Convention

### General Rules

Use hump nomenclature in C, C++, and ObjC, such as `CityCat` and `bigDoghouse`.

### Prefix

private, protected member variables, prefixed by `m`, such as `mCat`; global variables, class static variables, prefixed with `g`, such as `gWorld`; all non-static C functions, assembly functions need to be prefixed with `MNN`.

### Visibility

All functions and classes that need to be exposed should be marked with `MNN_PUBLIC`.


# Best Practices
## Near Release Principle

To reduce the risk of memory leaks, allocating temporary memory and freeing memory should be implemented in adjacent code blocks, ie, smart pointers or AutoStorage classes should be used.

## Defensive Programming

For external input, you should verify its validity, such as:

``` C
MNN_PUBLIC struct MNNNet* MNNCreateNet(const char* path) {
    if (NULL == path) {
        MNN_PRINT("input path is NULL, failed to create net!\n");
        return NULL;
    }

    // ...
}
```

For internal input, you should use `MNN_ASSERT` to avoid problem codes, such as:

``` C
void copyFloats(const float* input, float* output, int size) {
    MNN_ASSERT(NULL != input);
    MNN_ASSERT(NULL != output);
    for (int i = 0; i < size; ++i) {
        output[i] = input[i];
    }
}
```

It is forbidden to ignore errors without comments, such as:

``` C
void setUnitDimensions(const int* dims, int size) {
    if (NULL == dims) return; // should not directly return without comments

    for (int i = 0; i < size; ++i) {
        dims[i] = 1;
    }
}
```

## Notes

For all non-Op header files:
1. class needs to be commented with the purpose of the class.
2. All *non-override* methods need to be commented with its usage, the purpose of each parameter, and the return value information (if any).
3. All public member variables (generally members of the structure) need to be commented with their purpose.

Example:
``` C
/**
 * @brief function description
 * @param param param description
 * @return return value description
 */
int example(int param) {
    // ...
}
```

# Special Restrictions
## C++

For reasons of performance analysis, except the introduced three-party code, MNN code must follow:
1. prohibits operator overloading of classes.
2. prohibits the implementation of copy constructor, overload assignment operator of classes.
3. prohibits custom constructors of structs.

For reasons of controlling the size of the library file, in addition to the introduced three-party code, the MNN code is subject to:
1. Streams are not allowed, such as cout/cin, ifstream/ofstream, istringstream/ostringstream, etc.
2. C++ exception is not allowed, ie try/catch/throw.

## Assembly

For cross-platform compilation, MNN code needs to follow:
1. All assembly needs to provide C equivalent implementation, assembly implementation or C language implementation is chosen according to build environment at compile time.
2. The parameter and return types must be 32-bit/64-bit compatible types, that is, one of pointer, size_t, and ssize_t. Disable other types to avoid calling convention mismatch between different compilation environments.
3. Use the register according to the ARM standard manual strictly. For example, the q4 - q7 on armv7a must be restored after use. The v8 - v15 on armv8 must be restored after use.
