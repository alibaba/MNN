# 代码风格
## 格式化工具
我们使用`clang-format`和`git-clang-format`统一C、C++、Objective-C代码风格。代码风格使用工程根目录下的`.clang-format`文件描述。

在新增文件时，使用clang-format调整代码风格：
```shell
clang-format -i /path/to/new_file
```

在修改文件时，使用git-clang-format调整新增部分的代码风格：
```shell
cd /path/to/MNN
git-clang-format
```

## 代码风格
对于C、C++、Objective-C代码，我们使用Google代码风格。但是对下列项目作出调整：

| 项目 | 修改 |
| --- | --- |
| AccessModifierOffset 访问权限修正偏移 | 由-1改为-4 |
| AlignConsecutiveAssignments 连续赋值对齐 | 由false改为true |
| ColumnLimit 列宽限制 | 由80改为120 |
| IndentWidth 缩进宽度 | 由2改为4 |
| ObjCBlockIndentWidth ObjC Block缩进宽度 | 由2改为4 |
| ObjCSpaceAfterProperty ObjC属性后保留空格 | 由false改为true |
| SpacesBeforeTrailingComments 行尾注释前空格数 | 由2改为1 |

## 命名约定
### 一般规则
在C、C++和ObjC中使用驼峰命名法，如`CityCat`和`bigDoghouse`。

### 前缀
private、protected的成员变量，以`m`为前缀，如`mCat`；全局变量、类静态变量，以`g`为前缀，如`gWorld`；所有非static C函数、汇编函数都需要以`MNN`为前缀。

### 可见性
所有需要对外暴露的函数、类，都需要使用`MNN_PUBLIC`标记。

## 最佳实践
### 临近释放原则
为降低内存泄露风险，申请临时内存和释放内存宜在相邻代码块内实现，即，应使用智能指针或AutoStorage类。

### 防御式编程
对于外部入参，应明确判定入参有效性，如：
```c
MNN_PUBLIC struct MNNNet* MNNCreateNet(const char* path) {
    if (NULL == path) {
        MNN_PRINT("input path is NULL, failed to create net!\n");
        return NULL;
    }
    // ...
}
```

对于内部入参，宜使用`MNN_ASSERT`避免问题代码的产生，如：
```c
void copyFloats(const float* input, float* output, int size) {
    MNN_ASSERT(NULL != input);
    MNN_ASSERT(NULL != output);
    for (int i = 0; i < size; ++i) {
        output[i] = input[i];
    }
}
```

禁止在没有注释适当理由的情况下，忽略错误入参，如：
```c
void setUnitDimensions(const int* dims, int size) {
    if (NULL == dims) return; // should not directly return without comments
    for (int i = 0; i < size; ++i) {
        dims[i] = 1;
    }
}
```

## 注释
对于所有非Op头文件：
1. class需要注释说明类的用途；
2. 所有_非override_的_public_方法需要通过注释说明方法、各参数的用途和返回值信息（若有）；
3. 所有public成员变量（一般为结构体成员）需说明其用途；

示例：
```c
/**
 * @brief function description
 * @param param param description
 * @return return value description
 */
int example(int param) {
    // ...
}
```

## 特殊限制
### C++
出于便于性能分析的理由，除引入的三方代码外，MNN代码需遵循：
1. class禁止运算符重载
2. class禁止实现拷贝构造函数、重载赋值运算符
3. struct禁止自定义构造函数


出于控制库文件大小的理由，除引入的三方代码外，MNN代码需遵循：
1. 不允许使用stream，如cout/cin、 ifstream/ofstream、 istringstream/ostringstream等
2. 不允许使用C++异常机制，即try/catch/throw

### 汇编
出于跨平台编译的诉求，MNN代码需遵循：
1. 所有汇编都需要有C语言等价实现，编译时，通过宏选择平台对应的汇编实现或C语言实现
2. 入参、返回值类型必须是32位/64位兼容类型，即指针、size_t、ssize_t之一，禁用其他类型，避免编译环境不同导致调用规约偏差
3. 严格按照ARM标准手册使用寄存器，如armv7a上q4 - q7使用后必须复原，armv8上v8 - v15 使用后必须复原
