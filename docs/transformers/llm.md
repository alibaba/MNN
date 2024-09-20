# 大语言模型

基于MNN开发的LLM推理引擎，支持目前主流的开源LLM模型。该功能分为2部分：
- 模型导出：将torch模型导出为onnx，然后转换为mnn模型；导出tokenizer文件，embedding等文件；
- 模型推理：支持导出的模型推理，支持LLM模型的文本生成；

## 模型导出

`llmexport`是一个llm模型导出工具，能够将llm模型导出为onnx和mnn模型。

### 用法
1. 将需要导出的LLM项目clone到本地，如：Qwen2-0.5B-Instruct
```sh
git clone https://www.modelscope.cn/qwen/Qwen2-0.5B-Instruct.git
```
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
4. `llm.mnn.json`: mnn模型对应的json文件，apply_lora或者gptq量化权重时使用；
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
     ├── llm.onnx
     ├── llm_config.json
     └── tokenizer.txt
```

### 功能
- 支持将模型为onnx或mnn模型，使用`--export onnx`或`--export mnn`
- 支持对模型进行对话测试，使用`--test $query`会返回llm的回复内容
- 默认会使用onnx-slim对onnx模型进行优化，跳过该步骤使用`--skip_slim`
- 支持合并lora权重后导出，指定lora权重的目录使用`--lora_path`
- 制定量化bit数使用`--quant_bit`；量化的block大小使用`--quant_block`
- 使用`--lm_quant_bit`来制定lm_head层权重的量化bit数，不指定则使用`--quant_bit`的量化bit数
- 支持使用自己编译的`MNNConvert`，使用`--mnnconvert`

### 参数
```
usage: llmexport.py [-h] --path PATH [--type TYPE] [--lora_path LORA_PATH] [--dst_path DST_PATH] [--test TEST] [--export EXPORT]
                    [--skip_slim] [--quant_bit QUANT_BIT] [--quant_block QUANT_BLOCK] [--lm_quant_bit LM_QUANT_BIT]
                    [--mnnconvert MNNCONVERT]

llm_exporter

options:
  -h, --help            show this help message and exit
  --path PATH           path(`str` or `os.PathLike`):
                        Can be either:
                        	- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]
                        	- A path to a *directory* clone from repo like `../chatglm-6b`.
  --type TYPE           type(`str`, *optional*):
                        	The pretrain llm model type.
  --lora_path LORA_PATH
                        lora path, defaut is `None` mean not apply lora.
  --dst_path DST_PATH   export onnx/mnn model to path, defaut is `./model`.
  --test TEST           test model inference with query `TEST`.
  --export EXPORT       export model to an onnx/mnn model.
  --skip_slim           Whether or not to skip onnx-slim.
  --quant_bit QUANT_BIT
                        mnn quant bit, 4 or 8, default is 4.
  --quant_block QUANT_BLOCK
                        mnn quant block, default is 0 mean channle-wise.
  --lm_quant_bit LM_QUANT_BIT
                        mnn lm_head quant bit, 4 or 8, default is `quant_bit`.
  --mnnconvert MNNCONVERT
                        local mnnconvert path, if invalid, using pymnn.
```

## 模型推理

### 编译

[从源码编译](../compile/other.html#id4)
在原有编译过程中增加必需编译宏即可： -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true 

- mac / linux / windows

以 mac / linux 为例 :
```
make build
cd build
cmake ../ -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true
make -j16
```

x86架构额外加 MNN_AVX512 的宏：
```
make build
cd build
cmake ../ -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_AVX512=true
make -j16
```

- Android：额外增加 MNN_ARM82 的宏
```
cd project/android
mkdir build_64
../build_64.sh "-DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_ARM82=true"
```

- iOS: 参考 transformers/llm/engine/ios/README.md
```
sh package_scripts/ios/buildiOS.sh "-DMNN_ARM82=true -DMNN_LOW_MEMORY=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_BUILD_LLM=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true"
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
  - visual_model: 当使用VL模型时，visual_model的实际路径为`base_dir + visual_model`，默认为`base_dir + 'visual.mnn'`
- 推理配置
  - max_new_tokens: 生成时最大token数，默认为`512`
  - reuse_kv: 多轮对话时是否复用之前对话的`kv cache`，默认为`false`
  - quant_qkv: CPU attention 算子中`query, key, value`是否量化，可选为：`0, 1, 2, 3, 4`，默认为`0`，含义如下：
    - 0: key和value都不量化
    - 1: 使用非对称8bit量化存储key
    - 2: 使用fp8格式量化存储value
    - 3: 使用非对称8bit量化存储key，使用fp8格式量化存储value
    - 4: 量化kv的同时使用非对称8bit量化query，并使用int8矩阵乘计算Q*K
  - use_mmap: 是否使用mmap方式，在内存不足时将权重写入磁盘，避免溢出，默认为false，手机上建议设成true
  - kvcache_mmap: 是否使用mmap方式，在内存不足时将在KV Cache 写入磁盘，避免溢出，默认为false
  - tmp_path: 启用 mmap 相关功能时，写入磁盘的缓存目录
    - iOS 上可用如下语句创建临时目录并设置：`NSString *tempDirectory = NSTemporaryDirectory();llm->set_config("{\"tmp_path\":\"" + std::string([tempDirectory UTF8String]) + "\"}")`
- 硬件配置
  - backend_type: 推理使用硬件后端类型，默认为：`"cpu"`
  - thread_num: CPU推理使用硬件线程数，默认为：`4`; OpenCL推理时使用`68`
  - precision: 推理使用精度策略，默认为：`"low"`，尽量使用`fp16`
  - memory: 推理使用内存策略，默认为：`"low"`，开启运行时量化

##### 配置文件示例
- `config.json`
  ```json
  {
      "llm_model": "qwen2-1.5b-int4.mnn",
      "llm_weight": "qwen2-1.5b-int4.mnn.weight",

      "backend_type": "cpu",
      "thread_num": 4,
      "precision": "low",
      "memory": "low"
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

#### GPTQ权重加载
- 使用脚本生成GPTQ模型权重，用法参考: [apply_gptq.py](../tools/script.html#apply-gptq-py)
- 创建`gptq.json`配置文件
  ```json
  {
      "llm_model": "model.mnn",
      "llm_weight": "gptq.mnn.weight",
  }
  ```


#### LoRA权重加载
- 使用脚本生成lora模型，用法参考: [apply_lora.py](../tools/script.html#apply-lora-py)
- lora模型使用
  - 直接加载lora模型使用，创建`lora.json`配置文件
  ```json
  {
      "llm_model": "lora.mnn",
      "llm_weight": "base.mnn.weight",
  }
  ```
  - 运行时选择并切换lora模型
  ```cpp
  // 创建并加载base模型
  std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
  llm->load();
  // 使用同一个对象，在多个lora模型之间选择性使用，不可以并发使用
  {
      // 在基础模型的基础上添加`lora_1`模型，模型的索引为`lora_1_idx`
      size_t lora_1_idx = llm->apply_lora("lora_1.mnn");
      llm->response("Hello lora1"); // 使用`lora_1`模型推理
      // 添加`lora_2`模型，并使用
      size_t lora_2_idx = llm->apply_lora("lora_2.mnn");
      llm->response("Hello lora2"); // 使用`lora_2`模型推理
      // 通过索引选择`lora_1`作为llm对象当前使用的模型
      llm->select_module(lora_1_idx);
      llm->response("Hello lora1"); // 使用`lora_1`模型推理
      // 释放加载的lora模型
      llm->release_module(lora_1_idx);
      llm->release_module(lora_2_idx);
      // 选择使用基础模型
      llm->select_module(0);
      llm->response("Hello base"); // 使用`base`模型推理
  }
  // 使用多个对象，可以并发的加载使用多个lora模型
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
