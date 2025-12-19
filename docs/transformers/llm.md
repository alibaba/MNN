# 大语言模型

基于MNN开发的LLM推理引擎，支持目前主流的开源LLM模型。该功能分为2部分：
- 模型导出：将torch模型导出为onnx，然后转换为mnn模型；导出tokenizer文件，embedding等文件；
- 模型推理：支持导出的模型推理，支持LLM模型的文本生成；

## 快速开始

### **第一步：模型导出 (Export)**

此步骤是将原始的 PyTorch 模型（如 Qwen2 系列）转换为 MNN 引擎可以加载和推理的格式。

1.  **安装依赖**：
    进入导出工具目录并安装必要的 Python 包。
    ```bash
    cd ./transformers/llm/export
    pip install -r requirements.txt
    ```

2.  **准备原始模型**：
    将需要部署的开源 LLM 模型（例如 `Qwen2-0.5B-Instruct`）克隆到本地。**务必确保 `git lfs` 已安装，以下载完整的模型文件**。
    ```bash
    git lfs install
    git clone https://www.modelscope.cn/qwen/Qwen2-0.5B-Instruct.git
    ```

3.  **执行导出命令**：
    运行 `llmexport.py` 脚本，将模型、Tokenizer、Embedding 等导出为 MNN 格式。
    ```bash
    python llmexport.py \
        --path /path/to/Qwen2-0.5B-Instruct \
        --export mnn --hqq
    ```
    *   **关键产物**：脚本会生成一个包含 `llm.mnn`, `llm.mnn.weight`, `tokenizer.txt`, `embeddings_bf16.bin`【可能存在】, `llm_config.json`, `config.json` 等文件的模型目录。

4.  **（可选）高级功能**：
    *   **量化**：通过 `--quant_bit 4` 和 `--quant_block 128` 等参数可以调节量化的Bits数，默认为`4 bit , block size 64`。通过 `--hqq` 或 `--awq` 可以启用对应算法以提升量化后的模型精度，一般建议增加`--hqq`
    *   **LoRA**：通过 `--lora_path` 合并或分离 LoRA 权重。
    *   **Embeding**：对于目前主流的8b以下模型，采用了`Tie-Embeding`技术，默认不会导出`embeddings_bf16.bin`，而是复用`llm.mnn.weight`中的`lm`权重，需要提升embed精度可以设置 `--seperate_embed` 分离出`embeddings_bf16.bin`。
    *   **GPTQ**：通过 `--gptq_path` 应用预量化好的 GPTQ 权重。
    *   **手动转换**：如果直接导出 `mnn` 失败，或者需要fp16/fp32精度的模型，可先导出 `onnx`，再用 `MNNConvert` 工具手动转换。

---

### **第二步：引擎编译 (Compile)**

此步骤是编译 MNN 的 C++ 推理引擎，使其支持 LLM 推理功能。

1.  **配置编译选项**：
    在标准的 MNN 编译命令中，**必须添加 `-DMNN_BUILD_LLM=true`** 以启用 LLM 支持。
    *   **Omni 模型**：如果需要支持图像/音频输入，还需添加 `-DMNN_BUILD_LLM_OMNI=ON`。
    *   **平台优化**：
        *   **x86 (Mac/Linux)**：可添加 `-DMNN_AVX512=true` 以利用 AVX512 指令集加速。
        *   **Android**：可添加 `-DMNN_OPENCL=true` 以利用 GPU 加速。
        *   **iOS**：可添加 `-DMNN_METAL=ON` 以利用 GPU 加速。
        *   **Web (WASM)**：使用 `emcmake` 并配置 `-DMNN_FORBID_MULTI_THREAD=ON` 等特定选项。

2.  **执行编译**：
    以 Linux/Mac 为例：
    ```bash
    mkdir build && cd build
    cmake .. -DMNN_BUILD_LLM=true -DMNN_AVX512=true # 根据平台调整选项
    make -j16
    ```
    编译完成后，会生成核心库文件（如 `libMNN.so`, `libllm.so`）。

---

### **第三步：运行时配置与推理 (Inference)**

此步骤是配置模型运行参数并启动推理。

1.  **准备模型目录**：
    将第一步导出的所有文件（`llm.mnn`, `llm.mnn.weight`, `tokenizer.txt`, `embeddings_bf16.bin`, `llm_config.json`）放在同一个文件夹下。

2.  **配置 `config.json`**：
    编辑或使用自动生成的 `config.json` 文件，根据你的硬件和需求调整参数：
    *   **硬件**：设置 `backend_type` (如 `"cpu"`, `"opencl"`) 和 `thread_num`。
    *   **性能**：设置 `precision` (如 `"low"` for fp16) 和 `memory` (如 `"low"` for runtime quant)。
    *   **生成**：设置 `max_new_tokens`, `sampler_type` (如 `"mixed"`), `temperature`, `topK`, `topP` 等。
    *   **高级**：设置 `reuse_kv` (多轮对话), `chunk` (内存分块) 等。
    *   **示例**：
        ```json
        {
            "backend_type": "cpu",
            "thread_num": 4,
            "precision": "low",
            "sampler_type": "mixed",
            "temperature": 0.7,
            "topP": 0.9,
            "reuse_kv": true
        }
        ```

3.  **运行推理 Demo**：
    使用编译好的 `llm_demo` 工具进行推理。
    *   **交互式聊天**：
        ```bash
        ./llm_demo /path/to/model_dir/config.json
        ```
    *   **批量处理 Prompt**：
        ```bash
        ./llm_demo /path/to/model_dir/config.json /path/to/prompt.txt
        ```
    *   **多模态输入** (Omni 模型)：在 Prompt 中嵌入 `<img>` 或 `<audio>` 标签。

4.  **（可选）性能基准测试**：
    使用 `llm_bench` 工具对不同后端、线程数、Prompt 长度等配置进行性能压测，以找到最优配置。
    ```bash
    ./llm_bench -m ./model/config.json -a cpu,opencl -t 4,8 -p 32,64 -n 32 -rep 3
    ```

---

**总结流程图**：
`准备PyTorch模型` -> `使用 llmexport.py 导出为 MNN 格式` -> `编译 MNN 引擎 (启用 LLM)` -> `配置 config.json` -> `使用 llm_demo 进行推理`



## 模型导出工具`llmexport`


`llmexport`是一个llm模型导出工具，能够将llm模型导出为onnx和mnn模型。

### 依赖安装
```
cd ./transformers/llm/export
pip install -r requirements.txt
```

### 用法
1. 将需要导出的LLM项目clone到本地，如：Qwen2-0.5B-Instruct
```sh
git lfs install
git clone https://www.modelscope.cn/qwen/Qwen2-0.5B-Instruct.git
```

***clone 后检查一下模型大小，有可能因为lfs没安装导致下载的是空模型***

3. 执行`llmexport.py`导出模型
```sh
cd ./transformers/llm/export
# 导出模型，tokenizer和embedding，并导出对应的mnn模型
python llmexport.py \
        --path /path/to/Qwen2-0.5B-Instruct \
        --export mnn
```
4. 导出产物
导出产物为：
1. `config.json`: 模型运行时的配置，可手动修改；
2. `embeddings_bf16.bin`: 模型的embedding权重二进制文件，推理时使用；
3. `llm.mnn`: 模型的mnn文件，推理时使用；
4. `llm.mnn.json`: mnn模型对应的json文件，`apply_lora`或gptq量化权重时使用；
5. `llm.mnn.weight`: 模型的mnn权重，推理时使用；
6. `llm.onnx`: 模型的onnx文件，不包含权重，推理时不使用；
7. `llm_config.json`: 模型的配置信息，推理时使用；
8. `tokenizer.txt`: 模型的tokenzier文件，推理时使用；
目录结构如下所示：
```
.
└── model
     ├── config.json
     ├── embeddings_bf16.bin
     ├── llm.mnn
     ├── llm.mnn.json
     ├── llm.mnn.weight
     ├── onnx/
          ├──llm.onnx
          ├──llm.onnx.data
     ├── llm_config.json
     └── tokenizer.txt
```

### 功能
- 直接转为mnn模型，使用`--export mnn`，注意，你需要先安装pymnn或者通过`--mnnconvert`选项指定MNNConvert工具的地址，两种条件必须满足其中一个。如果没有安装pymnn并且没有通过`--mnnconvert`指定MNNConvert工具的地址，那么llmexport.py脚本会在目录"../../../build/"下寻找MNNConvert工具，需保证该目录下存在MNNConvert文件。此方案目前支持导出4bit和8bit模型
- 如果直接转为mnn模型遇到问题，或者需要其他bits数的量化（如5bit/6bit），可以先将模型先转为onnx模型，使用`--export onnx`，然后使用./MNNConvert工具将onnx模型转为mnn模型:

```
./MNNConvert --modelFile ../transformers/llm/export/model/onnx/llm.onnx --MNNModel llm.mnn --keepInputFormat --weightQuantBits=4 --weightQuantBlock=128 -f ONNX --transformerFuse=1 --allowCustomOp --saveExternalData
```

- 支持对模型进行对话测试，使用`--test $query`会返回llm的回复内容
- 支持合并lora权重后导出，指定lora权重的目录使用`--lora_path`
- 制定量化bit数使用`--quant_bit`；量化的block大小使用`--quant_block`
- 使用`--lm_quant_bit`来制定lm_head层权重的量化bit数，不指定则使用`--quant_bit`的量化bit数

### 参数
执行 `python llmexport.py -h` 可查看参数：
```
usage: llmexport.py [-h] --path PATH [--type TYPE] [--tokenizer_path TOKENIZER_PATH] [--lora_path LORA_PATH]
                    [--gptq_path GPTQ_PATH] [--dst_path DST_PATH] [--verbose] [--test TEST] [--export EXPORT]
                    [--onnx_slim] [--quant_bit QUANT_BIT] [--quant_block QUANT_BLOCK]
                    [--lm_quant_bit LM_QUANT_BIT] [--mnnconvert MNNCONVERT] [--ppl] [--awq] [--sym] [--seperate_embed]
                    [--lora_split]

llm_exporter

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           path(`str` or `os.PathLike`):
                        Can be either:
                            - A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]
                            - A path to a *directory* clone from repo like `../chatglm-6b`.
  --type TYPE           type(`str`, *optional*):
                            The pretrain llm model type.
  --tokenizer_path TOKENIZER_PATH
                        tokenizer path, defaut is `None` mean using `--path` value.
  --lora_path LORA_PATH
                        lora path, defaut is `None` mean not apply lora.
  --gptq_path GPTQ_PATH
                        gptq path, defaut is `None` mean not apply gptq.
  --dst_path DST_PATH   export onnx/mnn model to path, defaut is `./model`.
  --verbose             Whether or not to print verbose.
  --test TEST           test model inference with query `TEST`.
  --export EXPORT       export model to an onnx/mnn model.
  --onnx_slim           Whether or not to use onnx-slim.
  --quant_bit QUANT_BIT
                        mnn quant bit, 4 or 8, default is 4.
  --quant_block QUANT_BLOCK
                        mnn quant block, 0 mean channle-wise, default is 128.
  --visual_quant_bit VISUAL_QUANT_BIT
                        mnn visual model quant bit, 4 or 8, default is setting in utils/vision.py by different vit model.
  --visual_quant_block VISUAL_QUANT_BLOCK
                        mnn visual model quant block, 0 mean channle-wise, default is setting in utils/vision.py by different vit model.
  --lm_quant_bit LM_QUANT_BIT
                        mnn lm_head quant bit, 4 or 8, default is `quant_bit`.
  --mnnconvert MNNCONVERT
                        local mnnconvert path, if invalid, using pymnn.
  --ppl                 Whether or not to get all logits of input tokens.
  --awq                 Whether or not to use awq quant.
  --sym                 Whether or not to using symmetric quant (without zeropoint), defualt is False.
  --visual_sym          Whether or not to using symmetric quant (without zeropoint) for visual model, defualt is False.
  --seperate_embed      For lm and embed shared model, whether or not to sepearte embed to avoid quant, defualt is False, if True, embed weight will be seperate to embeddingbf16.bin.
  --lora_split          Whether or not export lora split, defualt is False.
```


### 权重读取
llmexport.py 同时支持 LLM 的验证功能，有较多的依赖。在没有相应环境的情况下，MNN-LLM也提供由 safetensors 或 gguf 文件读取权重的工具，可以降低内存需求，提高转换速度。使用方法如下：

#### 权重读取前置工作
1. 下载模型结构：在如下地址找到对应的MNN模型并下载（建文件夹 model，单独下载4个文件： llm.mnn , llm_config.json, tokenizer.txt , config.json）
```
https://modelscope.cn/organization/MNN
```

2. 安装 pymnn ，并把 llm.mnn 转换成 llm.mnn.json
```
pip install MNN
mnnconvert -f MNN --modelFile model/llm.mnn --JsonFile model/llm.mnn.json
```

#### safetensors 转 mnn

使用 safetensors2mnn.py 读取权重：

```
python3 safetensors2mnn.py --path /Users/xtjiang/.cache/modelscope/hub/Qwen/Qwen2___5-0___5B-Instruct --mnn_dir model
```

safetensors2mnn.py 支持设定量化参数，和 llmexport.py 一致

#### gguf 转 mnn
使用 gguf2mnn.py 读取 gguf 文件

```
python3 gguf2mnn.py --gguf ~/third/llama.cpp/build/ggml-model-Q4_K.gguf --mnn_dir model
```

目前本方案不支持多模态的模型转换。


## 模型推理

### 编译

[从源码编译](../compile/other.html#id4)
在原有编译过程中增加llm开关即可：
```
-DMNN_BUILD_LLM=ON
```

若需要开启Omni功能（支持图像/音频输入），增加`MNN_BUILD_LLM_OMNI`选项
```
-DMNN_BUILD_LLM=ON -D MNN_BUILD_LLM_OMNI=ON
```

#### mac / linux / windows

以 mac / linux 为例 :
```
make build
cd build
cmake ../ -DMNN_BUILD_LLM=true
make -j16
```

x86架构额外加 `MNN_AVX512` 的宏：
```
make build
cd build
cmake ../ -DMNN_BUILD_LLM=true -DMNN_AVX512=true
make -j16
```

#### Android：额外增加`MNN_OPENCL`的宏
```
cd project/android
mkdir build_64
../build_64.sh -DMNN_BUILD_LLM=true -DMNN_OPENCL=true -DMNN_USE_LOGCAT=true
```
高通设备部分视觉模型支持NPU功能，可增加`MNN_QNN`宏启用QNN功能。QNN运行分2种模式：
- 在线编译QNN模型：运行其它后端统一的mnn模型，运行时进行编译构图，通过需要较长的构图启动时间，主要用于功能正确性验证。
- 离线编译QNN模型：使用MNN2QNNModel转换工具将统一的mnn模型离线编译转换成含有Plugin算子的mnn模型以及QNN模型，运行时直接运行编译好的QNN模型，用于生产部署情况。此时需要开启`MNN_WITH_PLUGIN`宏。
```
cd project/android
mkdir build_64
../build_64.sh -DMNN_BUILD_LLM=true -DMNN_OPENCL=true -DMNN_QNN=true -DMNN_WITH_PLUGIN=true -DMNN_USE_LOGCAT=true
```

#### iOS: 参考 transformers/llm/engine/ios/README.md
```
sh package_scripts/ios/buildiOS.sh -DMNN_BUILD_LLM=true
```

#### Web
环境配置参考 https://mnn-docs.readthedocs.io/en/latest/compile/engine.html#web

- 编译库，产出 `libMNN.a`，`libMNN_Express.a`，`libllm.a`

```
mkdir buildweb
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -DMNN_FORBID_MULTI_THREAD=ON -DMNN_USE_THREAD_POOL=OFF -DMNN_USE_SSE=OFF -DMNN_BUILD_LLM=true
make -j16
```

- Demo 编译

```
emcc ../transformers/llm/engine/demo/llm_demo.cpp -I ../include -I ../transformers/llm/engine/include libMNN.a libllm.a express/libMNN_Express.a -o llm_demo.js --preload-file ~/qwen2.0_1.5b/ -s ALLOW_MEMORY_GROWTH=1 -o llm_demo.js
```

使用如下命令测试：
```
node llm_demo.js ~/qwen2.0_1.5b/config.json ~/qwen2.0_1.5b/prompt.txt
```

### 使用
#### 运行时配置

##### 运行时文件
将导出产物中用于模型推理的部分置于同一个文件夹下，添加一个配置文件`config.json`来描述模型名称与推理参数，目录如下：
```
.
└── model_dir
     ├── config.json
     ├── embeddings_bf16.bin
     ├── llm_config.json
     ├── llm.mnn
     ├── llm.mnn.weight
     └── tokenizer.txt
```

##### 配置项
配置文件支持以下配置：
- 模型文件信息
  - base_dir: 模型文件加载的文件夹目录，默认为config.json的所在目录，或模型所在目录；
  - llm_config: `llm_config.json`的实际名称路径为`base_dir + llm_config`，默认为`base_dir + 'config.json'`
  - llm_model: `llm.mnn`的实际名称路径为`base_dir + llm_model`，默认为`base_dir + 'llm.mnn'`
  - llm_weight: `llm.mnn.weight`的实际名称路径为`base_dir + llm_weight`，默认为`base_dir + 'llm.mnn.weight'`
  - block_model: 分段模型时`block_{idx}.mnn`的实际路径为`base_dir + block_model`，默认为`base_dir + 'block_{idx}.mnn'`
  - lm_model: 分段模型时`lm.mnn`的实际路径为`base_dir + lm_model`，默认为`base_dir + 'lm.mnn'`
  - embedding_model: 当embedding使用模型时，embedding的实际路径为`base_dir + embedding_model`，默认为`base_dir + 'embedding.mnn'`
  - embedding_file: 当embedding使用二进制时，embedding的实际路径为`base_dir + embedding_file`，默认为`base_dir + 'embeddings_bf16.bin'`
  - tokenizer_file: `tokenizer.txt`的实际名称路径为`base_dir + tokenizer_file`，默认为`base_dir + 'tokenizer.txt'`
  - visual_model: 当使用VL模型时，visual_model的实际路径为`base_dir + visual_model`，默认为`base_dir + 'visual.mnn'`、
  - audio_model: 当使用Audio模型时，audio_model的实际路径为`base_dir + audio_model`，默认为`base_dir + 'audio.mnn'`
  - Omni模型文件信息
    - talker_model: 当使用Omni模型时，talker_model的实际路径为`base_dir + talker_model`，默认为`base_dir + 'talker.mnn'`
    - talker_weight: 当使用Omni模型时，talker_weight的实际路径为`base_dir + talker_weight`，默认为`base_dir + 'talker.mnn.weight'`
    - talker_embedding_file: 当使用Omni模型时，talker_embedding_file的实际路径为`base_dir + talker_embedding_file`，默认为`base_dir + 'talker_embeddings_bf16.bin'`
    - predit_model: 当使用Omni模型时，predit_model的实际路径为`base_dir + predit_model`，默认为`base_dir + 'predit.mnn'`
    - dit_model: 当使用Omni模型时，dit_model的实际路径为`base_dir + dit_model`，默认为`base_dir + 'dit.mnn'`
    - bigvgan_model: 当使用Omni模型时，bigvgan_model的实际路径为`base_dir + bigvgan_model`，默认为`base_dir + 'bigvgan.mnn'`
    - spk_dict: 当使用Omni模型时，spk_dict的实际路径为`base_dir + spk_dict`，默认为`base_dir + 'spk_dict.txt'`
    - context_file: 配置上下文信息文件路径，实际路径为`base_dir + context_file`，默认`base_dir + 'context.json'`，内容格式为json格式的上下文信息，包含：如tools，enable_thinking等信息。
- 推理配置
  - max_new_tokens: 生成时最大token数，默认为`512`
  - reuse_kv: 多轮对话时是否复用之前对话的`kv cache`，默认为`false`.
  - quant_qkv: CPU attention 算子中`query, key, value`是否量化，可选为：`0, 1, 2, 3, 4`，默认为`0`，含义如下：
    - 0: key和value都不量化
    - 1: 使用非对称8bit量化存储key
    - 2: 使用fp8格式量化存储value
    - 3: 使用非对称8bit量化存储key，使用fp8格式量化存储value
    - 4: 量化kv的同时使用非对称8bit量化query，并使用int8矩阵乘计算Q*K
  - use_mmap: 是否使用mmap方式，在内存不足时将权重写入磁盘，避免溢出，默认为false，手机上建议设成true
  - chunk: 限制每次最大处理的token数，高于此值将分块运行，以减少内存占用，eg: chunk: 128
  - chunk_limits: 限制每次处理的token数，不在此范围内将分拆或者补零处理，eg: chunk_limits: [128, 1] , 存在 chunk_limits 时，chunk 配置无效
  - kvcache_mmap: 是否使用mmap方式，在内存不足时将在KV Cache 写入磁盘，避免溢出，默认为false
  - tmp_path: 启用 mmap 相关功能时，写入磁盘的缓存目录
    - iOS 上可用如下语句创建临时目录并设置：`NSString *tempDirectory = NSTemporaryDirectory();llm->set_config("{\"tmp_path\":\"" + std::string([tempDirectory UTF8String]) + "\"}")`
- 硬件配置
  - backend_type: 推理使用硬件后端类型，默认为：`"cpu"`
  - thread_num: CPU推理使用硬件线程数，默认为：`4`; OpenCL推理时使用`68`(不是传统意义的线程数，代表的是opencl buffer存储和tuning wide模式)
  - precision: 推理使用精度策略，默认为：`"low"`，尽量使用`fp16`
  - memory: 推理使用内存策略，默认为：`"low"`，开启运行时量化
- 与CPU动态量化相关的配置，提升精度、性能
  - dynamic_option: 推理时是否对feature map分blocksize/group进行量化。可选为：`0, 1, 2, 8, 9, 10`，默认是`0`，含义如下：
    - 0: feature map数据使用per channel量化
    - 1: feature map数据使用per tensor量化
    - 2: feature map数据用per block量化，blocksize等于权重量化时的blocksize，如果权重量化时没有使用per block量化，即使设置2，也不会对feature map做per block量化
    - 8+n(n=0,1,2): 该选项是为了加速LLM 推理时Decode性能。但是当prompt长度小于300时，Prefill速度会显著变慢。当prompt长度高于300时，Prefill速度不会变慢。
  - cpu_sme2_neon_division_ratio: 为了提高Arm SME后端多线程推理时性能，可根据模型、线程数定制化设置该参数。参数计算方式: Prefill阶段单个SME核和NEON核的工作量比例x:1，Decode阶段工作量比例y:1，
                                  则参数设置为8*x+y，x和y均是不大于7的正整数。41、49和33是常见的参数设置. 可以通过观察单线程推理时，SME后端相较于NEON后端的加速比来决定该参数的取值。默认是`41`.
- Sampler配置
  - sampler_type: 使用的sampler种类，目前支持`greedy`, `temperature`, `topK`, `topP`, `minP`, `tfs`, `typical`, `penalty`8种基本sampler，外加`mixed`(混合sampler，当选择`mixed`时，依次执行mixed_samplers中的sampler)。默认为`greedy`，但是建议使用`mixed`、`temperature`来增加输出多样性，或使用`penalty`来降低重复。
  - mixed_samplers: 当`sampler_type`为`mixed`时有效，默认为`["topK", "tfs", "typical", "topP", "min_p", "temperature"]`, 模型计算得到的logits会依次经过这些sampler采样。
  - temperature: `temperature`, `topP`, `minP`, `tfsZ`, `typical`中temerature值，默认为1.0
  - topK: `topK`中top K 个的个数，默认为40
  - topP: `topP`中top P的值，默认为0.9
  - minP: `minP`中min P的值，默认为0.1
  - tfsZ: `tfs`中Z的值，默认为1.0 (即不使用tfs算法)
  - typical: `typical`中p的值，默认为1.0 (即不使用typical算法)
  - penalty: `penalty`中对于logits中重复token的惩罚项，默认为0.0 (即不惩罚)，推荐值为1.05~1.5。
  - n_gram: 最大存储的ngram大小，超过此大小的重复ngram将被禁止重复输出，仅在`penalty`选中时生效，默认为8
  - ngram_factor: `penalty`中对于重复ngram (n>1) 的额外惩罚，默认为1.0，即没有额外惩罚
  - penalty_sampler: `penalty`中施加完惩罚项后采用的sampling策略，可选"greedy"或"temperature"，默认greedy.
- 投机解码配置项
  - speculative_type: 投机解码算法设置，当前仅支持配置为`lookahead`(使用外接知识库/输入prompt信息去生成草稿做投机验证),通常需要较完备的知识库或者输入prompt与输出重合度较高的场景(例如：代码编辑、文本总结)才有较明显加速。
  - draft_predict_length: 草稿长度，通常设置2-8之间，默认为4。
  - draft_match_strictness: 草稿匹配的严格程度，当有草稿时，是否选取该草稿去做并行验证。可以设置`low`、`medium`、`high`。通常严格程度越高，草稿接受率越高，但是启用并行验证概率也越低。默认为`low`，该参数仅`lookahead`模式设置有效。
  - draft_selection_rule: 草稿选择规则，当有多个草稿时，选取的规则设置。支持`freqxlen`（出现频率与匹配长度最高者）和`fcfs`(最先匹配者)。默认`freqxlen`，该参数仅`lookahead`模式设置有效。
  - ngram_match_maxlen: ngram匹配历史token最长值，默认为4，该参数仅`lookahead`模式设置有效。
  - lookup_file: 用户外接知识库文件路径，默认为`lookup_file.txt`，该参数仅`lookahead`模式设置有效。
  - ngram_update: 是否解码过程实时添加更新ngram信息，默认为`false`，该参数仅`lookahead`模式设置有效。
- Omni语音生成配置
  - talker_max_new_tokens: 生成时最大语音token数，在Qwen2.5-Omni中50个语音token对应1秒语音，默认为`2048`
  - talker_speaker: 生成语音的音色，Qwen2.5-Omni中支持的音色为：`["Chelsie", "Ethan"]`
  - dit_steps: 生成语音时扩散模型迭代次数，默认为`5`, 建议设置为`5~10`, 越大语音质量越高计算耗时越高；
  - dit_solver: 生成语音时扩散模型求解算法阶数，支持`1, 4`，默认为`1`使用一阶欧拉法；`4`表示四阶龙格库塔法，效果略好但耗时增加4倍；

##### 配置文件示例
- `config.json`
  ```json
  {
      "llm_model": "qwen2-1.5b-int4.mnn",
      "llm_weight": "qwen2-1.5b-int4.mnn.weight",

      "backend_type": "cpu",
      "thread_num": 4,
      "precision": "low",
      "memory": "low",
      "sampler_type": "mixed",
      "mixed_samplers": ["topK", "tfs", "typical", "topP", "min_p", "temperature"],
      "temperature": 1.0,
      "topK": 40,
      "topP": 0.9,
      "tfsZ": 1.0,
      "minP": 0.1,
      "reuse_kv": true
  }
  ```
- `llm_config.json`
  ```json
  {
      "hidden_size": 1536,
      "layer_nums": 28,
      "attention_mask": "float",
      "key_value_shape": [
          2,
          1,
          0,
          2,
          128
      ],
      "prompt_template": "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
      "is_visual": false,
      "is_single": true
  }
  ```
- `context.json`
  ```json
  {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "获取当前时间"
            }
        }
    ],
    "enable_thinking": false
  }
  ```

#### 推理用法
`llm_demo`的用法如下：
```
# 使用config.json
## 交互式聊天
./llm_demo model_dir/config.json
## 针对prompt中的每行进行回复
./llm_demo model_dir/config.json prompt.txt

# 不使用config.json, 使用默认配置
## 交互式聊天
./llm_demo model_dir/llm.mnn
## 针对prompt中的每行进行回复
./llm_demo model_dir/llm.mnn prompt.txt
```

- 对于视觉大模型，在prompt中嵌入图片输入
```
<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>介绍一下图片里的内容
# 指定图片大小
<img><hw>280, 420</hw>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>介绍一下图片里的内容
```
- 对于音频大模型，在prompt中嵌入音频输入
```
<audio>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav</audio>介绍一下音频里的内容
```

#### 单个模型对话性能测评
建议使用config.json, 可以自行配置运行后端、线程数、输出token数限制等选项。
```
## 注意：当选择opencl后端时，thread_num需设为68。
## 注意：测评opencl后端性能时，由于第一次运行会tuning生成缓存文件(性能较慢)，因此需要运行第二次(已经有缓存文件)来看性能数据。

./llm_demo model_dir/config.json prompt.txt
```

#### LLM Benchmark工具使用
使用llm_bench可以比较不同模型在不同配置下的性能差异。

##### llm_bench参数列表
```
usage: ./llm_bench [options]

options:
  -h, --help
  -m, --model <filename>                    (default: ./Qwen2.5-1.5B-Instruct)
  -a, --backends <cpu,opencl,metal>         (default: cpu)
  -c, --precision <n>                       (default: 2) | Note: (0:Normal(for cpu bakend, 'Nornal' is 'High'),1:High,2:Low)
  -t, --threads <n>                         (default: 4)
  -p, --n-prompt <n>                        (default: 512)
  -n, --n-gen <n>                           (default: 128)
  -pg <pp,tg>                               (default: 512,128)
  -mmp, --mmap <0|1>                        (default: 0)
  -rep, --n-repeat <n>                      (default: 5)
  -kv, --kv-cache <true|false>              (default: false) | Note: if true: Every time the LLM model generates a new word, it utilizes the cached KV-cache
  -fp, --file-print <stdout|filename>       (default: stdout) ｜ If not 'stdout', all test results will be written to the specified file.
```

##### llm_bench 参数解释
- '-m | --model': llm.mnn和llm.mnn.weight文件所在的文件夹中config.json文件的路径，而不是文件夹的路径或者mnn/mnn.weight文件的路径。 可以填写多个模型的config.json文件地址，使用英文逗号分隔；
- '-a | --backends': 指定运行LLM模型的后端，目前MNN仅支持CPU/METAL/OPENCL后端。可以填写多个后端，后端名称均使用英文小写字母，使用英文逗号分隔；
- '-t | --threads': 指定CPU后端推理时采用的线程数。对于OPENCL后端，该字段表示的不是线程数，而是GPU MODE，当前LLM推理时OpenCL均采取Buffer模式推理，线程数设置为4时性能较优。对于METAL后端对性能的影响较小。可以填写多个线程数，使用英文逗号分隔；
- '-p | --n-prompt': 指定推理时处理的prompt长度，可填写多个长度，使用英文逗号分隔；测试结果表示LLM模型的首字符响应速度；
- '-n | --n-gen': 指定推理时生成字符的长度，可填写多个长度，使用英文逗号分隔；测试结果表示LLM模型在不考虑历史KV信息时生成一个字符的速度，即Attention算子中past_kv_length=0;
- '-pg': 指定prompt长度和生成字符数量，测试中该项的耗时是前两项('-p'和'-n')耗时的总和，处理字符的数量是prompt长度和生成字符数量之和；
- '-mmp | --mmap': 指定模型加载时是否使用mmap技术，只能填写一个候选项，0或1；该项对模型推理性能无影响；
- '-rep | --n-repeat': 每一个测试实例重复的次数，最终结果取平均数，并计算性能的标准差；
- '-kv | --kv-cache': 当设置为true时，测试时在LLM模型decode阶段会考虑历史KV信息，即测试方法和运行'llm_demo'程序一致；
- '-fp | --file-print': 默认输出到屏幕上；如果指定了输出文件，最终的测试结果会以追加的方式以markdown格式写入到文件中，不会删除文件中已有的内容；文件不存在会自动创建。

##### 命令行运行llm_bench
在build目录下运行
```bash
./llm_bench -m ./Qwen2.5-1.5B-Instruct/config.json,./Qwen2.5-0.5B-Instruct/config.json -a cpu,opencl,metal -c 1,2 -t 8,12 -p 16,32 -n 10,20 -pg 8,16 -mmp 0 -rep 4 -kv true -fp ./test_result
```

#### 多Prompt场景下KVCache选择性复用
rollback_demo提供了多Prompt场景下自行选择复用部分kvcache的示例代码。
```bash
./rollback_demo /path/to/model_dir/config.json /path/to/prompt.txt <cache_prefix_in_disk> <max_token_number>
```
其中，prompt.txt需要包含至少三组prompt。
- cache_prefix_in_disk需要设置为0或1。
- cache_prefix_in_disk 设置1表示：第一段Prompt是后续Prompt的公共前缀Prompt，第二、三段Prompt分别是基于第一段Prompt后续的文本内容。第一次启动会将前缀Prompt的KVCache缓存在磁盘文件中。第二次启动会跳过公共前缀Prompt的Prefill，直接在磁盘中加载，提升Prefill速度。。
- cache_prefix_in_disk 设置0表示：在多段Prompt下，如何删除不需要的KVCache，仅保留关联性的KVCache示例。

#### GPTQ权重
需要使用GPTQ权重，可以在导出模型时，使用`--gptq_path PATH`来指定的路径，使用如下：
```bash
# 导出GPTQ量化的模型
python llmexport.py --path /path/to/Qwen2.5-0.5B-Instruct --gptq_path /path/to/Qwen2.5-0.5B-Instruct-GPTQ-Int4 --export mnn
```

#### LoRA权重
LoRA权重有两使用方式：1. 合并LoRA权重到原始模型；2. LoRA模型单独导出。

第一种模式速度更快，使用更简单但是不支持运行时切换；第二种略微增加一些内存和计算开销，但是更加灵活，支持运行时切换LoRA，适合多LoRA场景。
##### 融合LoRA

将LoRA权重合并到原始模型中导出，在模型导出时指定`--lora_path PATH`参数，默认使用合并方式导出，使用如下：
```bash
# 导出LoRA合并的模型
python llmexport.py --path /path/to/Qwen2.5-0.5B-Instruct --lora_path /path/to/lora --export mnn
```

融合LoRA模型使用与原始模型使用方法完全一样。

##### 分离LoRA

将LoRA单独导出为一个模型，支持运行时切换，在模型导出时指定`--lora_path PATH`参数，并指定`--lora_split`，就会将LoRA分离导出，使用如下：
```bash
python llmexport.py --path /path/to/Qwen2.5-0.5B-Instruct --lora_path /path/to/lora --lora_split --export mnn
```
导出后模型文件夹内除了原始模型外，还会增加`lora.mnn`，这个就是lora模型文件。

运行时创建lora模型
  ```cpp
  // 创建并加载base模型
  std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
  llm->load();
  // 创建lora模型，支持多个lora模型并存，支持并发
  {
      std::mutex creat_mutex;
      auto chat = [&](const std::string& lora_name) {
          MNN::BackendConfig bnConfig;
          auto newExe = Executor::newExecutor(MNN_FORWARD_CPU, bnConfig, 1);
          ExecutorScope scope(newExe);
          Llm* current_llm = nullptr;
          {
              std::lock_guard<std::mutex> guard(creat_mutex);
              current_llm = llm->create_lora(lora_name);
          }
          current_llm->response("Hello");
      };
      std::thread thread1(chat, "lora_1.mnn");
      std::thread thread2(chat, "lora_2.mnn");
      thread1.join();
      thread2.join();
  }
  ```

#### 获取语音输出
使用Omni模型时，可以使用接口`setWavformCallback`获取语音输出，使用接口`generateWavform`开始输出语音。
注意`setWavformCallback`需要在文本生成前调用， `generateWavform`在文本生成结束后调用，示例如下：

1. 保存语音到文件中
```cpp
#include <audio/audio.hpp>
int main() {
  // save wavform to file for debug
  std::vector<float> waveform;
  llm->setWavformCallback([&](const float* ptr, size_t size, bool last_chunk) {
      waveform.reserve(waveform.size() + size);
      waveform.insert(waveform.end(), ptr, ptr + size);
      if (last_chunk) {
          auto waveform_var = MNN::Express::_Const(waveform.data(), {(int)waveform.size()}, MNN::Express::NCHW, halide_type_of<float>());
          MNN::AUDIO::save("output.wav", waveform_var, 24000);
          waveform.clear();
      }
      return true;
  });
  llm->response("Hello");
  // generate wavform
  llm->generateWavform();
  return 0;
}

```
2. 流式播放语音（Mac/iOS为例）
```cpp
#include <thread>
#include <AudioToolbox/AudioToolbox.h>

struct AudioPlayer {
    AudioStreamBasicDescription format;
    std::vector<float> audioBuffer;
    std::mutex bufferMutex;
    std::condition_variable bufferCondVar;
    bool doneGenerating = false;
    std::thread playThread;
    AudioPlayer() {
        format.mSampleRate = 24000;
        format.mFormatID = kAudioFormatLinearPCM;
        format.mFormatFlags = kLinearPCMFormatFlagIsFloat;
        format.mBytesPerPacket = sizeof(float);
        format.mFramesPerPacket = 1;
        format.mBytesPerFrame = sizeof(float);
        format.mChannelsPerFrame = 1;
        format.mBitsPerChannel = sizeof(float) * 8;
    }
    bool play(const float* ptr, size_t size, bool last_chunk);
};

void AudioQueueCallback(void* userData, AudioQueueRef inAQ, AudioQueueBufferRef inBuffer) {
    AudioPlayer* context = static_cast<AudioPlayer*>(userData);
    std::unique_lock<std::mutex> lock(context->bufferMutex);
    int samplesToCopy = inBuffer->mAudioDataBytesCapacity / sizeof(float);
    while (context->audioBuffer.size() < samplesToCopy) {
        if (context->doneGenerating) { break; }
        context->bufferCondVar.wait(lock);
    }
    if (context->audioBuffer.size() < samplesToCopy) {
        samplesToCopy = context->audioBuffer.size();
    }
    memcpy(inBuffer->mAudioData, context->audioBuffer.data(), samplesToCopy * sizeof(float));
    context->audioBuffer.erase(context->audioBuffer.begin(), context->audioBuffer.begin() + samplesToCopy);
    inBuffer->mAudioDataByteSize = samplesToCopy * sizeof(float);
    AudioQueueEnqueueBuffer(inAQ, inBuffer, 0, nullptr);
}

void playAudioData(AudioPlayer* context) {
    AudioQueueRef queue;
    AudioQueueNewOutput(&context->format, AudioQueueCallback, context, nullptr, nullptr, 0, &queue);
    AudioQueueBufferRef buffers[3];
    UInt32 bufferSize = 1024 * sizeof(float);
    for (int i = 0; i < 3; ++i) {
        AudioQueueAllocateBuffer(queue, bufferSize, &buffers[i]);
        AudioQueueCallback(context, queue, buffers[i]);
    }
    AudioQueueStart(queue, nullptr);
    while (true) {
        {
            std::lock_guard<std::mutex> lock(context->bufferMutex);
            if (context->doneGenerating && context->audioBuffer.empty())
                break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    AudioQueueStop(queue, true);
    for (int i = 0; i < 3; ++i) {
        AudioQueueFreeBuffer(queue, buffers[i]);
    }
    AudioQueueDispose(queue, true);
}

bool AudioPlayer::play(const float* ptr, size_t size, bool last_chunk) {
    {
        std::lock_guard<std::mutex> lock(bufferMutex);
        audioBuffer.reserve(audioBuffer.size() + size);
        audioBuffer.insert(audioBuffer.end(), ptr, ptr + size);
    }
    if (playThread.joinable()) {
        bufferCondVar.notify_all();
    } else {
        playThread = std::thread(playAudioData, this);
        printf(">>>>>>>> PLAY START\n");
    }
    if (last_chunk) {
        doneGenerating = true;
        bufferCondVar.notify_all();
        if (playThread.joinable()) {
            playThread.join();
            printf(">>>>>>>> PLAY END\n");
        }
        return false;
    }
    return true;
}

int main() {
    //....
    AudioPlayer audio_player;
    llm->setWavformCallback([&](const float* ptr, size_t size, bool last_chunk) {
        return audio_player.play(ptr, size, last_chunk);
    });
    //....
    llm->response("Hello");
    // generate wavform
    llm->generateWavform();
    return 0;
}
```

### Python 中使用
参考 `pymnn/examples/MNNLlm` 下面的 demo 使用

```
import MNN.llm as llm
import sys

if len(sys.argv) < 2:
    print('usage: python llm_example.py <path_to_model_config>')
    exit(1)

config_path = sys.argv[1]
# create model
qwen = llm.create(config_path)
# load model
qwen.load()

# response stream
out = qwen.response('你好', True)
print(out)

out_ids = qwen.generate([151644, 872, 198, 108386, 151645, 198, 151644, 77091])
print(out_ids)
```

## NPU 推理 LLM

使用NPU推理，需要特定的导出参数，并针对目标设备转换出相应的模型。目前支持使用高通芯片和MTK芯片的NPU进行推理。一般流程是：LLM模型导出->转换成对应设备NPU模型->推到目标设备运行

### LLM 模型导出
NPU运行LLM需要特定的量化格式，需要按如下参数以导出 mnn
`--smooth --act_bit=16 --quant_block=0 --lm_quant_bit=16 --quant_bit=4 --seperate_embed --sym --act_sym`

eg:
```
python3 llmexport.py --path /Users/xtjiang/.cache/modelscope/hub/models/Qwen/Qwen3-4B --export mnn --smooth --act_bit=16 --quant_block=0 --lm_quant_bit=16 --seperate_embed --quant_bit=4 --sym --act_sym
```

### QNN LLM

#### 获得QNN依赖

可通过以下步骤获取依赖：
- [注册高通账号](https://myaccount.qualcomm.com/signup)
- 访问Qualcomm AI Engine Direct SDK（即QNN SDK），下载SDK，并解压。比如`/home/xiaying/third/qnn/qairt/2.38.0.250901`
- 修改`~/.bashrc` ，增加SDK路径到环境变量, 然后运行 `source ~/.bashrc` 或者重启终端。eg：

```
export QNN_SDK_ROOT=/home/xiaying/third/qnn/qairt/2.38.0.250901
export QNN_ROOT=/home/xiaying/third/qnn/qairt/2.38.0.250901
export HEXAGON_SDK_ROOT=/home/xiaying/third/qnn/qairt/2.38.0.250901
```

#### 构建 QNN 模型

在模型转换器编译时，增加`-DMNN_QNN=ON -DMNN_QNN_CONVERT_MODE=ON`，eg:

```
cd ${MNN_ROOT}
mkdir build && cd build
cmake .. -DMNN_QNN=ON -DMNN_QNN_CONVERT_MODE=ON -DMNN_BUILD_TOOLS=ON -DMNN_BUILD_LLM=ON
make -j16
```


使用 `npu/generate_llm_qnn.py` 构建 qnn 模型
eg:

```
cd ${MNN_ROOT}
cd transformers/llm/export
python3 npu/generate_llm_qnn.py --model model --soc_id=57 --dsp_arch=v75
```

目标设备`soc_id` 和 `dsp_arch` 可在高通官方查询，如下为一些设备的参考

| 硬件    | SOC ID | HEXAGON ARCH |
| :------ | :----- | :----------- |
| 8 Gen 1 | 36     | 69           |
| 8 Gen 2 | 43     | 73           |
| 8 Gen 3 | 57     | 75           |
| 8 Elite | 69     | 79           |


***执行成功后，会在 model 目录下产出 config_qnn.json 及 model/qnn 目录***

***构建完成后，model 目录下的 llm.mnn 及 llm.mnn.weight 不再需要，可以删除以减少文件总大小***

#### Android设备上运行QNN LLM

- 编译 MNN Android 库并推送到目标设备，编译时需要增加 `-DMNN_QNN=ON -DMNN_WITH_PLUGIN=ON`，eg:

```
cd ${MNN_ROOT}
cd project/android
mkdir build_64 && cd build_64
../build_64.sh -DMNN_QNN=ON -DMNN_WITH_PLUGIN=ON -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON
../updateTest.sh
```

- 参考如下脚本把 QNN 相关 so 放到 Android 对应测试目录中

```
ANDROID_WORKING_DIR=/data/local/tmp/MNN/
HEXAGON_ARCH=v75
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${ANDROID_WORKING_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV${HEXAGON_ARCH}Stub.so ${ANDROID_WORKING_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v${HEXAGON_ARCH}/unsigned/libQnnHtpV${HEXAGON_ARCH}Skel.so ${ANDROID_WORKING_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so ${ANDROID_WORKING_DIR}
```

- 推送模型并执行

推送模型：
```
cd ${MNN_ROOT}
cd transformers/llm/export
adb push model /data/local/tmp/MNN/model
```

运行：
```
cd ${MNN_ROOT}
project/android/testCommon.sh ./llm_demo model/config_qnn.json
```

### MTK LLM
#### 获得 MTK SDK
- 目前MTK没有开放SDK获得方案，需自行联系MTK取得支持，获得对应的SDK
- 获取后，修改`~/.bashrc`，添加环境变量，eg:

```
export NEURON_SDK=/home/xiaying/third/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk
```

#### 构建 MLDA 模型
MLDA 是 MTK 的 NPU 推理引擎，需要把 MNN 模型转成 MLDA 模型才可在其NPU上运行

- 增加MNN对应的预转换后端配置 `-DMNN_NEUROPILOT=ON` ，eg:

```
cd ${MNN_ROOT}
mkdir build && cd build
cmake ../ -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_LLM=ON -DMNN_NEUROPILOT=ON
make -j4
```

- 确定设备的`mlda`版本号和编译选项，并修改`source/backend/neuropilot/npu_convert.py`的`archoptions`，当前默认配置为`--arch=mdla5.1 --l1-size-kb=7168 --num-mdla=4`，支持天玑9300的NPU编译

- 使用 `npu/generate_llm_mlda.py` 构建 MLDA 模型

```
cd ${MNN_ROOT}
cd transformers/llm/export
python3 npu/generate_llm_mlda.py --model model
```

执行成功后，会在 model 目录下产出`config_mlda.json`与`mlda`目录。

***生成后，原先的llm.mnn和llm.mnn.weight可以删除***

#### Android设备上运行 MLDA LLM

- 增加`-DMNN_NEUROPILOT=ON -DMNN_WITH_PLUGIN=ON`编译 MNN Android 库

```
cd ${MNN_ROOT}
cd project/android/
mkdir build_64
cd build_64
../build_64.sh -DMNN_NEUROPILOT=ON -DMNN_WITH_PLUGIN=ON -DMNN_BUILD_LLM=ON
../updateTest.sh
```

- 推送模型并执行

推送模型：
```
cd ${MNN_ROOT}
cd transformers/llm/export
adb push model /data/local/tmp/MNN/model
```

运行：
```
cd ${MNN_ROOT}
project/android/testCommon.sh ./llm_demo model/config_mlda.json
```
