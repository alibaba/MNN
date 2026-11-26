# 知合 A210 TORQ delegate

MNN 负责分词、采样和流式输出，TORQ 在 A210 NPU 上执行预编译文本图的 prefill 和 decode。已验证 Qwen3.5-2B Q5、batch 1。TORQ SDK 2.1.5 在每个请求的 prefill 开始时清空 KV cache 和线性注意力状态。

提示和生成的 token 共用编译图的 `max_seq_len`。请求生成的数量超过剩余容量时，TORQ 会打印请求值和截断后的值。提示恰好占满容量时会生成 0 个 token，状态为 `MAX_TOKENS_FINISHED`。

## 准备模型

在导出机上准备同一模型的 Hugging Face 目录和 TORQ 编译产物。下例将 Qwen3.5-2B 的文本图放在 `lm-640-c32/`，其中包含 `torq.llm.json`、`torq.bm.json` 及清单引用的图和权重。图需提供独立的 `linear_output` 和 BF16 logits。知合的 0.8B 示例说明了量化和编图流程。[TORQ Model Zoo 编图说明](https://gitee.com/zhcomputing/torq-model-zoo/blob/e55875e553ad11c5229b63ad4c0db608819569ed/examples/qwen3_5/README.md)

在导出机的 MNN 仓库根目录安装导出依赖并生成运行配置。

```sh
python3 -m pip install -r transformers/llm/export/requirements.txt
python3 transformers/llm/export/llmexport.py --path /path/to/Qwen3.5-2B \
  --export torq --torq_model_dir ./lm-640-c32 --dst_path ./lm-640-c32/mnn
```

导出器使用 `--path` 下的分词器，以匹配编译图的 token ID，并在 `mnn/config.json` 中写入 `"delegate": "TORQ"`。配置用相对路径引用原图，部署时将整个 `lm-640-c32/` 复制到板上。

## 在 A210 上构建

在板上加载 SDK 运行环境，将下面两个路径换成各自 SDK 的 `a210` 目录，再从 MNN 仓库根目录构建。

```sh
TORQ_SDK=/path/to/torq-llm-2.1.5/a210
UCA_SDK=/path/to/uca-nn-2.1.7/a210
export LD_LIBRARY_PATH="$TORQ_SDK/lib:$UCA_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S . -B build -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON \
  -DMNN_LLM_DELEGATES=ON -DMNN_LLM_TORQ=ON \
  -DTORQ_INCLUDE="$TORQ_SDK/include/torq-llm" -DUCA_INCLUDE="$UCA_SDK/include/uca-nn" \
  -DTORQ_LIBRARY="$TORQ_SDK/lib/libtorq_llm.so"
cmake --build build --target llm_demo llm_bench -j2
```

## 运行与测速

将模型目录放在板上当前目录下。第一条命令启动交互对话，输入 `/exit` 退出后可运行第二条测速。

```sh
./build/llm_demo ./lm-640-c32/mnn/config.json
./build/llm_bench -m ./lm-640-c32/mnn/config.json -p 0 -n 0 -pg 512,128 -rep 3
```

测速时确认 `backend` 列显示 `TORQ`。`-pg 512,128` 在同一请求内先处理 512 个提示 token，再接着生成 128 个 token。模型加载后预热 1 轮、统计 3 轮，Qwen3.5-2B Q5 在 A210 上的一次实测均值为 prefill 228.98 tok/s、decode 13.98 tok/s。
