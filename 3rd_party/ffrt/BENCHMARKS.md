# Benchmarks

## 测试场景

1. `fork_join`：通过构造 fork/join 执行时间，模拟线程创建和堵塞场景；
2. `fib`：通过构造斐波那契序列，模拟嵌套调用场景；
3. `face_story`：通过构造人脸数据，模拟人脸检测场景；

## 测试方法

```shell
cd benchmarks
./benchmarks count  # count 表明执行次数
```

## 测试结果

1. 测试数据和分析归档到 `benchmarks/output/tag_${stamp}/benchmark_${stamp}.svg`，其中 `stamp` 是最近一次 commit 提交时间
2. 测试结果已取平均
