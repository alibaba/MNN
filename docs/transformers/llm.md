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
```
usage: llmexport.py [-h] --path PATH [--type TYPE] [--lora_path LORA_PATH] [--dst_path DST_PATH] [--test TEST] [--export EXPORT]
                    [--quant_bit QUANT_BIT] [--quant_block QUANT_BLOCK] [--lm_quant_bit LM_QUANT_BIT]
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
在原有编译过程中增加必需编译宏即可：
```
-DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true
```

- 需要开启视觉功能时，增加相关编译宏
```
-DLLM_SUPPORT_VISION=true -DMNN_BUILD_OPENCV=true -DMNN_IMGCODECS=true
```
- 需要开启音频功能时，增加相关编译宏
```
-DLLM_SUPPORT_AUDIO=true -DMNN_BUILD_AUDIO=true
```

#### mac / linux / windows

以 mac / linux 为例 :
```
make build
cd build
cmake ../ -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true
make -j16
```

x86架构额外加 `MNN_AVX512` 的宏：
```
make build
cd build
cmake ../ -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_AVX512=true
make -j16
```

#### Android：额外增加 `MNN_ARM82` 和`MNN_OPENCL`的宏
```
cd project/android
mkdir build_64
../build_64.sh "-DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_ARM82=true -DMNN_OPENCL=true -DMNN_USE_LOGCAT=true"
```

#### iOS: 参考 transformers/llm/engine/ios/README.md
```
sh package_scripts/ios/buildiOS.sh "-DMNN_ARM82=true -DMNN_LOW_MEMORY=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_BUILD_LLM=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true"
```

#### Web
环境配置参考 https://mnn-docs.readthedocs.io/en/latest/compile/engine.html#web

- 编译库，产出 `libMNN.a`，`libMNN_Express.a`，`libllm.a`

```
mkdir buildweb
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-msimd128 -msse4.1" -DMNN_FORBID_MULTI_THREAD=ON -DMNN_USE_THREAD_POOL=OFF -DMNN_USE_SSE=ON -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true
make -j16
```

- Demo 编译

```
emcc ../transformers/llm/engine/llm_demo.cpp -DCMAKE_CXX_FLAGS="-msimd128 -msse4.1" -I ../include -I ../transformers/llm/engine/include libMNN.a libllm.a express/libMNN_Express.a -o llm_demo.js --preload-file ~/qwen2.0_1.5b/ -s ALLOW_MEMORY_GROWTH=1 -o llm_demo.js
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

#### GPTQ权重
需要使用GPTQ权重，可以在导出[Qwen2.5-0.5B-Instruct]模型时，使用`--gptq_path PATH`来指定[Qwen2.5-0.5B-Instruct-GPTQ-Int4]()的路径，使用如下：
```bash
# 导出GPTQ量化的模型
python llmexport.py --path /path/to/Qwen2.5-0.5B-Instruct --gptq_path /path/to/Qwen2.5-0.5B-Instruct-GPTQ-Int4 --export mnn
```

#### LoRA权重
LoRA权重有两使用方式：1. 合并LoRA权重到原始模型；2. LoRA模型单独导出。

第一种模式速度更快，使用更简单但是不支持运行时切换；第二种略微增加一些内存和计算开销，但是更加灵活，支持运行时切换LoRA，适合多LoRA场景。
##### 融合LoRA

###### 导出
将LoRA权重合并到原始模型中导出，在模型导出时指定`--lora_path PATH`参数，默认使用合并方式导出，使用如下：
```bash
# 导出LoRA合并的模型
python llmexport.py --path /path/to/Qwen2.5-0.5B-Instruct --lora_path /path/to/lora --export mnn
```

###### 使用
融合LoRA模型使用与原始模型使用方法完全一样。

##### 分离LoRA

###### 导出
将LoRA单独导出为一个模型，支持运行时切换，在模型导出时指定`--lora_path PATH`参数，并指定`--lora_split`，就会将LoRA分离导出，使用如下：
```bash
python llmexport.py --path /path/to/Qwen2.5-0.5B-Instruct --lora_path /path/to/lora --lora_split --export mnn
```
导出后模型文件夹内除了原始模型外，还会增加`lora.mnn`，这个就是lora模型文件。

###### 使用
- lora模型使用
  - 直接加载lora模型使用，创建`lora.json`配置文件，这样与直接运行融合LoRA的模型相似。
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
