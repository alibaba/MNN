# §3 并发 / 线程竞争（host 侧：引用计数、共享生命周期）

> **归属**：[`general-debug`](SKILL.md) 的分类分册之一，先在入口的分流表确认类别再读本文。
>
> **不在本文**：GPU 侧「逐次结果不同」的竞争（单 dispatch 内 threadgroup 竞争 / 融合别名）见
> [`memory-aliasing.md` §1.6](memory-aliasing.md#16-参考案例融合引入的别名竞争layernorm-折进-conv1x12026-08-03)；
> CPU 多线程动态分发 × 异构 kernel 导致的逐 run 数值差异见 [`nondeterminism.md`](nondeterminism.md) §10；
> 纯崩溃（栈、信号、真机 crash 日志的读法）见 [`crash-debug`](../crash-debug/SKILL.md)。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

**触发**（满足以下之一强烈怀疑本类）：
- 崩溃点在**析构链**上（`~XxxMemObj` / `~Tensor` / 容器析构），但那段代码单线程跑一万次都没问题；
- 栈顶在**后台线程 / 队列**上销毁对象（iOS 的 GCD `_dispatch_*`、Android 线程池），业务侧"用完就异步释放"；
- 线上偶现、本地必不复现；崩溃地址不合法或像被复用过的堆内存。

## 3.1 核心心法

**崩溃点 ≠ 根因点。** 引用计数被写坏时不会当场崩，等到某次析构真正 `free` 才炸，
所以栈上出现的类往往只是**最后一个持有者**。看到析构链崩溃 + 多线程释放，
先查**共享所有权的计数本身**（`RefCount` / `SharedPtr` / `shared_ptr` 的自定义替代品），
而不是崩溃那个类的内存池。

判断"是否可能跨线程释放"的关键是找到**同一对象被两个所有者持有**的证据：
grep 谁把内部对象（如 tensor 的 `mem` / `InsideDescribe`）**共享给了用户侧句柄**，
以及是否有 `use_count() > 1` 之类的分支。只要存在这种共享，就存在两个线程各自释放的可能。

## 3.2 定位手法：把概率问题变成确定信号

用 `git show <修复前commit>:<文件>` 取出旧实现，与当前实现各编一份**最小复现程序**，
在 **ThreadSanitizer** 下跑同一段并发释放逻辑做 A/B：

```bash
# 旧实现应报 data race；新实现应 0 告警
clang++ -std=c++11 -O1 -g -fsanitize=thread repro.cpp -I<old_header_dir> -o repro_old
clang++ -std=c++11 -O1 -g -fsanitize=thread repro.cpp -I<new_header_dir> -o repro_new
```

TSAN 会直接指出对同一地址的两次写，比反复跑真机等偶现快若干个量级。
**A/B 是必须的**——只跑新实现"没报错"不能证明修复有效，只能证明当前没竞争。

## 3.3 如何守住：UB 类竞争不能靠普通构建的运行时测试

⚠️ **最容易自欺的一步。** 非原子读改写在多线程下是 UB，编译器有权认定这对操作
无可观测副作用而**整段删除**。因此"写个多线程压力测试放进 CI"在普通 `-O2` 构建下
**必然假通过**：计数器从不漂移，测试永远绿。

判别方法很简单——**看耗时**。若"有 bug 版"跑同样百万级操作的耗时比修复版低一两个数量级，
说明循环被优化掉了，这个测试没有任何守护能力。

正确的分层：

| 守护 | 生效范围 | 能否抓住回退 |
|------|---------|------------|
| 不变量写成 `static_assert`（放在被保护的成员旁边，附**为什么**必须如此） | 每次编译、所有构建、零成本 | ✅ 确定性 |
| 并发运行时测试 | 仅 sanitizer 构建 | ✅ 确定性 |
| 并发运行时测试 | 普通 `-O2` 构建 | ❌ 假通过 |

两条都要落地，并且**运行时测试文件里必须写明"只在 TSAN 下有牙"**，
否则后人会以为普通 CI 已经覆盖了它。

## 3.4 通过标准

- 反向验证哨兵：把不变量改坏（如把原子计数改回普通 int），确认**编译真的失败**且报错信息能读懂；
- 反向验证测试：用真正的修复前代码跑，确认 TSAN **确实报竞争**；
- 正向验证：当前代码 0 告警，且新测试在普通构建里耗时可接受（可长期留在 CI）。

参见 `test/core/RefCountThreadSafetyTest.cpp` 与 `source/core/AutoStorage.h` 中
`RefCount::mNum` 旁的哨兵；新增测试的编译/结果判读陷阱见
[test-ci](../test-ci/test-suite.md) § "Adding a new test"。
