# 编译构建

## 构建依赖

因为当前 FFRT 编译依赖三方安全库（参见：[third_party_bounds_checking_function](https://gitee.com/openharmony/third_party_bounds_checking_function)）的功能，当进行 linux 构建时，需要把三方安全库的代码一并下载到本地，放置目录如下

```plain
├── resourceschedule_ffrt
├── third_party
    └── bounds_checking_function
```

## 构建参数

常见的构建宏定义说明

| 构建宏             | 含义                                                                                                                                                            |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FFRT_EXAMPLE       | 在 CMakeList 中用于控制是否编译 ffrt examples                                                                                                                   |
| FFRT_BENCHMARKS    | 在 CMakeList 中用于控制是否编译 ffrt benchmarks                                                                                                                 |
| FFRT_TEST_ENABLE   | 在 CMakeList 中用于控制是否编译 ffrt unittest cases                                                                                                             |
| FFRT_CLANG_COMPILE | 在 CMakeList 中用于控制是否使用 clang 编译                                                                                                                      |
| FFRT_SANITIZE      | 在 CMakeList 中用于控制是否开启 sanitizer 检测                                                                                                                  |
| FFRT_BBOX_ENABLE   | 在 CMakeList 中用于控制是否开启 FFRT 黑匣子，用于记录发生 crash 时 FFRT 线程、协程的状态                                                                        |
| FFRT_LOG_LEVEL     | 在 CMakeList&BUILD.gn 中用于动态设置 FFRT 默认日志级别，支持 0-3，依次为 ERROR，WARN，INFO，DEBUG，例如需要打开 DEBUG 及以上级别的日志，设置 `FFRT_LOG_LEVEL=3` |

## Linux 构建

### 编译环境

编译的 Linux 环境需要已安装 CMake（版本 3.10 以上）以及编译器 GCC 或 Clang

### 编译脚本

ffrt 代码根目录下的 CMakeList

### 编译命令

使用 CMake 编译：

```shell
mkdir -p build
cd build && cmake .. -DFFRT_EXAMPLE=ON
cmake --build . -j                      # add `-j` for parallel compilation
./examples/ffrt_submit                  # run some ffrt examples
```

或者直接使用代码仓中的 shell 脚本：

```shell
sh scripts/run_example.sh
```

### 编译产物

FFRT 样例的可执行文件 ffrt_submit

```plain
├── resourceschedule_ffrt
    └── build
        └──examples
           └──ffrt_submit
```

FFRT 动态库文件 libffrt.so

```plain
├── resourceschedule_ffrt
    └── build
        └──src
           └──libffrt.so
```

## OpenHarmony 构建

### 编译环境

编译环境部署参考 OpenHarmony 社区官方文档：[编译构建指导](https://gitee.com/openharmony/docs/blob/master/zh-cn/device-dev/subsystems/subsys-build-all.md)

### 编译脚本

ffrt 代码根目录下的 BUILD.gn

### 编译命令

在官方提供的产品编译命令中指定编译 FFRT 模块，仅需将 build-target 参数设置成 ffrt

以 rk3568 产品为例，进入源码根目录，执行：

64 位

```shell
./build.sh --product-name rk3568 --target-cpu arm64 --ccache --build-target ffrt
```

32 位

```shell
./build.sh --product-name rk3568 --ccache --build-target ffrt
```

### 开发依赖

```json
external_deps = [
  "ffrt:libffrt",
]
```
