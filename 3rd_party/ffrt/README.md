# 并发编程框架 FFRT

- [并发编程框架 FFRT](#并发编程框架-ffrt)
  - [简介](#简介)
  - [目录](#目录)
  - [编译构建](#编译构建)
  - [Testing](#testing)
  - [Benchmarks](#benchmarks)
  - [Release](#release)
  - [Contributing Code](#contributing-code)

## 简介

FFRT: Function Flow Runtime，一种并发编程框架，提供以数据依赖的方式构建异步并发任务的能力，包括数据依赖管理、任务执行器、系统事件处理等。并采用基于协程的任务执行方式，可以提高任务并行度、提升线程利用率、降低系统线程总数；充分利用多核平台的计算资源，保证系统对所有资源的集约化管理。最终解决系统线程资源滥用问题，打造极致用户体验。

功能介绍详见：[FFRT 用户指南](docs/README.md)

## 目录

```plain
├── benchmarks                  # 性能对比测试用例
├── docs                        # 用户指南
├── examples                    # 使用案例
├── interfaces                  # 接口目录
│   ├── inner_api               # 内部接口
│   └── kits                    # 对外接口
│       ├── c
│       └── cpp
├── scripts
├── src
│   ├── core                    # 核心模块
│   ├── dfx                     # 维测功能
│   ├── dm                      # 依赖管理
│   ├── eu                      # 执行单元
│   ├── internal_inc            # 对内接口目录
│   ├── ipc
│   ├── queue
│   ├── sched
│   ├── sync
│   ├── tm
│   └── util
├── test
└── tools
    └── ffrt_trace_process
```

## 编译构建

For detailed instructions on building FFRT, see [BUILD.md](BUILD.md).

## Testing

For detailed instructions on testing FFRT, see [TESTING.md](TESTING.md).

## Benchmarks

To evaluate and verify the performance of FFRT, extensive benchmark tests were conducted. These tests utilized various datasets and real-world scenarios to ensure reliability and efficiency across different environments.

Detailed test results and analyses can be found in the `benchmarks` directory. We encourage community members to perform further tests based on their specific use cases and provide feedback to help us continuously improve FFRT.

For detailed instructions on conducting benchmark tests, see [BENCHMARKS.md](BENCHMARKS.md).

## Release

Please see the [Release](RELEASE.md) note for information about C and C++ API versions.

## Contributing Code

Please see the [Contributing](CONTRIBUTING.md) guide for information on how to contribute code.
