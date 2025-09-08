# Testing FFRT

## XTS

XTS 介绍详见 OpenHarmony 社区官方文档：[XTS 子系统](https://gitee.com/openharmony/xts_acts/blob/master/README_zh.md)

### 代码路径

已覆盖 FFRT 所有 API 开源接口，用例代码详见：[FFRT 用例](https://gitee.com/openharmony/xts_acts/tree/master/resourceschedule/resourceschedule_standard/ffrt)

**用例 Hap 名称**：`ActsFfrtNativeTest`

### 编译依赖

参考 OpenHarmony 社区官方文档：[编译构建指导](https://gitee.com/openharmony/docs/blob/master/zh-cn/device-dev/subsystems/subsys-build-all.md)

### 编译命令

以 rk3568 产品为例，进入源码根目录，执行：

```shell
./build.sh --product-name rk3568 system_size=standard target_subsystem=resourceschedule
```

### 执行命令

```shell
run -l ActsFfrtNativeTest
```

## TDD

### 代码路径

```plain
foundation/resourceschedule/ffrt/test/
```

### 编译依赖

参考 OpenHarmony 社区官方文档：[编译构建指导](https://gitee.com/openharmony/docs/blob/master/zh-cn/device-dev/subsystems/subsys-build-all.md)

### 编译命令

以 rk3568 产品为例，进入源码根目录，执行：

64 位

```shell
./build.sh --product-name rk3568 --target-cpu arm64 --ccache --build-target foundation/resourceschedule/ffrt/test/ut:ffrt_unittest_ffrt
```

32 位

```shell
./build.sh --product-name rk3568 --ccache --build-target foundation/resourceschedule/ffrt/test/ut:ffrt_unittest_ffrt
```

### 执行命令

```shell
run -t ut -ss ffrttest
```

### Running unit-tests via cmake/ctest

```shell
cmake -S. -Bbuild -DFFRT_TEST_ENABLE=ON
cmake --build build --target ffrt_ut
ctest --test-dir build/test
```
