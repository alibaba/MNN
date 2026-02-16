# Sana Android Distill 模型集成与优化报告

本报告总结了 2026-01-20 Session 中针对 **Sana Distill 模型** 在 Android 端进行的性能优化、稳定性修复及功能增强工作。

---

## 1. 核心目标
1.  **性能飞跃**：将推理总耗时从约 50s 优化至 20s 以内。
2.  **显存稳定**：彻底解决 Android 端 OpenCL 后端的 OOM (Out of Memory) 和 Segfault 崩溃。
3.  **功能增强**：支持任意尺寸图片输入，自动 Resize 适配模型并在输出时还原。
4.  **自动化**：提供完备的增量推送与基准测试脚本。

---

## 2. 关键代码变更与引用

### A. 算法优化：移除 CFG (Classifier-Free Guidance)
**目的**：针对 Distill (蒸馏) 版本的 Sana 模型，移除不再需要的负向提示词计算，将核心推理 Batch Size 从 2 减小为 1。

**代码引用 1：LLM 处理逻辑简化 (Batch 2 -> 1)**
文件：`transformers/diffusion/engine/include/diffusion/sana_llm.hpp`
移除 `cfg_prompt` 参数，仅对单个提示词进行 Embedding 提取。

```cpp
// 变更：函数签名由 process(prompt, cfg_prompt) 改为 process(prompt)
VARP process(const std::string& prompt) {
    // ...
    std::vector<std::string> prompts;
    prompts.push_back(format_prompt(prompt)); // 仅处理正向提示词

    // Tokenize & Pad 后的 Batch 维度自动变为 1
    int batch = all_ids.size(); 
    
    // 移除对 meta_queries 的拼接复制
    auto meta_queries_batch = _Unsqueeze(meta_queries, {0}); 
    // ...
}
```

**代码引用 2：Diffusion 去噪循环优化**
文件：`transformers/diffusion/engine/src/diffusion.cpp`
移除 Latents 的复制操作，并精简了对 Transformer 模块的调用及噪声合成逻辑。

```cpp
// 变更：将 Mask 维度设为 1
VARP encoder_attention_mask = _Input({1, seq_len}, NCHW, halide_type_of<float>());

for (int i = 0; i < num_inference_steps; ++i) {
    // ...
    // 直接获取 Batch=1 的模型输出，无需进行 _Split 与引导计算
    auto res = mModules[2]->onForward({sample, prompt_embeds, timestep_var, encoder_attention_mask, ref_latents_batched});
    auto noise_pred_guided = _Convert(res[0], NCHW); 
    
    // 直接执行 Euler Step
    sample = sample + noise_pred_guided * _Const(dt / 1000.0f);
}
```

---

### B. 稳定性与显存管理 (稳定性修复)
**目的**：消除计算图在循环中的累积效应，并解决 OpenCL 后端在动态调整时的不稳定问题。

**代码引用 3：阻断计算图膨胀 (Detaching Graph)**
文件：`transformers/diffusion/engine/src/diffusion.cpp`
通过强制 `readMap` 并在每一步将数据 `memcpy` 到全新的 `_Input` 张量中，彻底阻断了隐式计算图的增长链。

```cpp
// 在每个 Step 结束时执行
{
    auto ptr = sample->readMap<float>(); // 强制执行当前图
    auto info = sample->getInfo();
    auto new_sample = _Input(info->dim, info->order, info->type); // 创建脱离依赖的新节点
    ::memcpy(new_sample->writeMap<float>(), ptr, info->size * sizeof(float)); 
    sample = new_sample; // 丢弃旧的 VARP，释放历史依赖
}
```

**代码引用 4：分阶段内存释放**
文件：`transformers/diffusion/engine/sana_diffusion_demo.cpp`
在获取 LLM 特征后立即销毁 LLM 实例，最大限度腾出 GPU 显存。

```cpp
// LLM 推理完成后
{
    auto ptr = llm_out->readMap<float>(); // 确保计算已同步回 CPU/内存
    sana_llm.reset(); // 立即释放 LLM 对象及其占用的显存/内存
    MNN_PRINT("LLM memory released.\n");
}
```

**代码引用 5：禁用不稳定优化项**
文件：`transformers/diffusion/engine/src/diffusion.cpp`
注释掉 `Session_Resize_Fix`，避免其在 Batch 维度变更时引发 Segfault。

```cpp
// Resize fix
for (auto& m : mModules) {
    // // if (m) m->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix); // 已注释
}
```

---

### C. 功能增强：任意分辨率支持
**目的**：解耦“用户图片尺寸”与“模型运行尺寸”。

**代码引用 6：自动前/后处理 Resize**
文件：`transformers/diffusion/engine/src/diffusion.cpp`

```cpp
// 1. 记录原始图片高度与宽度
VARP raw_image = imread(inputImagePath);
original_height = info->dim[0];
original_width = info->dim[1];

// 2. 强制 Resize 至模型所需的 512x512
image = resize(image, Size(mWidth, mHeight), ...); 

// ... 模型推理 ...

// 3. VAE 解码后，将结果 Resize 回原始分辨率
if (original_width > 0 && original_height > 0) {
    image = resize(image, Size(original_width, original_height), ...);
}
```

---

## 3. 自动化测试工具
*   **脚本名称**：`run_sana_benchmark_android.sh`
*   **特性**：
    *   **增量推送 (Smart Push)**：仅在设备缺失大权重文件时进行推送，大幅缩短测试准备时间。
    *   **性能打点**：结合 C++ 端的 `DemoTimer` 类，自动抓取并展示各阶段（Load LLM, Load Diff, Infer LLM, Infer Diff）的毫秒级耗时。

---

## 4. 最终性能数据 (Android OpenCL)

| 测试项 | 耗时 (ms) | 状态 |
| :--- | :--- | :--- |
| **Load LLM** | ~937 ms | 正常 |
| **Infer LLM (Batch=1)** | ~792 ms | **优化** |
| **Load Diffusion Weights** | ~122 ms | 正常 |
| **Infer Diffusion (5 steps)** | **16,536 ms** | **核心提升** |
| **总耗时** | **~18.5 s** | **达标** |

**结论**：
通过 **"No CFG (Batch 1)"** 与 **"Graph Detaching"** 的组合优化，在确保输出图片尺寸还原的同时，成功将总耗时压低至 20s 以内（相比优化前约 50s 提升了 **2.7 倍**），且程序在 Android 端运行极其稳健。
