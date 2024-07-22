# 大语言模型

基于MNN开发的LLM推理引擎，支持目前主流的开源LLM模型。该功能分为2部分：
- 模型导出：将torch模型导出为onnx，然后转换为mnn模型；导出tokenizer文件，embedding等文件；
- 模型推理：支持导出的模型推理，支持LLM模型的文本生成；

## 模型导出

`llm_export`是一个llm模型导出工具，能够将llm模型导出为onnx和mnn模型。

### 用法
1. 将需要导出的LLM项目clone到本地，如：Qwen2-0.5B-Instruct
```sh
git clone https://www.modelscope.cn/qwen/Qwen2-0.5B-Instruct.git
```
3. 执行`llm_export.py`导出模型
```sh
cd ./transformers/llm/export
# 导出模型，tokenizer和embedding，并导出对应的mnn模型
python llm_export.py \
        --type Qwen2-0_5B-Instruct \
        --path /path/to/Qwen2-0.5B-Instruct \
        --export \
        --export_token \
        --export_embed --embed_bin \
        --export_mnn
```
4. 导出产物
导出产物为：
1. `embeddings_bf16.bin`: 模型的embedding权重二进制文件，推理时使用；
2. `llm_config.json`: 模型的配置信息，推理时使用；
3. `llm.onnx`: 模型的onnx文件，推理时不使用；
4. `tokenizer.txt`: 模型的tokenzier文件，推理时使用；
5. `llm.mnn`: 模型的mnn文件，推理时使用；
6. `llm.mnn.weight`: 模型的mnn权重，推理时使用；
目录结构如下所示：
```
.
├── onnx
|    ├── embeddings_bf16.bin
|    ├── llm_config.json
|    ├── llm.onnx
|    └── tokenizer.txt
└── mnn
     ├── llm.mnn
     └── llm.mnn.weight
```

### 功能
- 支持将模型完整导出为一个onnx模型，使用`--export`
- 支持将模型分段导出为多个模型，使用`--export_split`
- 支持导出模型的词表到一个文本文件，每行代表一个token；其中token使用base64编码；使用`--export_verbose`
- 支持导出模型的Embedding层为一个onnx模型，使用`--export_embed`，同时支持bf16格式，使用`--embed_bf16`
- 支持分层导出模型的block，使用`--export_blocks`导出全部层；使用`--export_block $id`导出指定层
- 支持导出模型的lm_head层为一个onnx模型，使用`--export_lm`
- 支持导出多模态模型的visual模型为一个onnx模型，使用`--export_visual`
- 支持对模型进行对话测试，使用`--test $query`会返回llm的回复内容
- 支持在导出onnx模型后使用onnxruntime对结果一致性进行校验，使用`--export_test`
- 支持将tokenizer导出为文本文件，使用`--export_token`
- 支持将导出的onnx模型转换为mnn模型，默认转换为非对称4bit量化，使用`--export_mnn`
- 指定导出路径使用`--onnx_path`和`--mnn_path`
- 默认会使用onnx-slim对onnx模型进行优化，跳过该步骤使用`--skip_slim`
- 支持合并lora权重后导出，指定lora权重的目录使用`--lora_path`

### 参数
```
usage: llm_export.py [-h] --path PATH
                     [--type {chatglm-6b,chatglm2-6b,chatglm3-6b,codegeex2-6b,Qwen-7B-Chat,Qwen-1_8B-Chat,Qwen-1_8B,Qwen-VL-Chat,Qwen1_5-0_5B-Chat,Qwen1_5-1_8B-Chat,Qwen1_5-4B-Chat,Qwen1_5-7B-Chat,Qwen2-1_5B-Instruct,Baichuan2-7B-Chat,Llama-2-7b-chat-ms,Llama-3-8B-Instruct,internlm-chat-7b,TinyLlama-1_1B-Chat,Yi-6B-Chat,deepseek-llm-7b-chat,phi-2,bge-large-zh,lora}]
                     [--lora_path LORA_PATH] [--onnx_path ONNX_PATH] [--mnn_path MNN_PATH] [--export_mnn] [--export_verbose] [--export_test] [--test TEST] [--export] [--export_split] [--export_token]
                     [--export_embed] [--export_visual] [--export_lm] [--export_block EXPORT_BLOCK] [--export_blocks] [--embed_bin] [--embed_bf16] [--skip_slim]

llm_exporter

options:
  -h, --help            show this help message and exit
  --path PATH           path(`str` or `os.PathLike`):
                        Can be either:
                        	- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]
                        	- A path to a *directory* clone from repo like `../chatglm-6b`.
  --type {chatglm-6b,chatglm2-6b,chatglm3-6b,codegeex2-6b,Qwen-7B-Chat,Qwen-1_8B-Chat,Qwen-1_8B,Qwen-VL-Chat,Qwen1_5-0_5B-Chat,Qwen1_5-1_8B-Chat,Qwen1_5-4B-Chat,Qwen1_5-7B-Chat,Qwen2-1_5B-Instruct,Baichuan2-7B-Chat,Llama-2-7b-chat-ms,Llama-3-8B-Instruct,internlm-chat-7b,TinyLlama-1_1B-Chat,Yi-6B-Chat,deepseek-llm-7b-chat,phi-2,bge-large-zh,lora}
                        type(`str`, *optional*):
                        	The pretrain llm model type.
  --lora_path LORA_PATH
                        lora path, defaut is `None` mean not apply lora.
  --onnx_path ONNX_PATH
                        export onnx model path, defaut is `./onnx`.
  --mnn_path MNN_PATH   export mnn model path, defaut is `./mnn`.
  --export_mnn          Whether or not to export mnn model after onnx.
  --export_verbose      Whether or not to export onnx with verbose.
  --export_test         Whether or not to export onnx with test using onnxruntime.
  --test TEST           test model inference with query `TEST`.
  --export              export model to an `onnx` model.
  --export_split        export model split to some `onnx` models:
                        	- embedding model.
                        	- block models.
                        	- lm_head model.
  --export_token        export llm tokenizer to a txt file.
  --export_embed        export llm embedding to an `onnx` model.
  --export_visual       export llm visual model to an `onnx` model.
  --export_lm           export llm lm_head to an `onnx` model.
  --export_block EXPORT_BLOCK
                        export llm block [id] to an `onnx` model.
  --export_blocks       export llm all blocks to `onnx` models.
  --embed_bin           export embedding weight as bin file with dtype `bfloat16`
  --embed_bf16          using `bfloat16` replace `float32` in embedding.
  --skip_slim           Whether or not to skip onnx-slim.
```

## 模型推理

### 编译

[从源码编译](../compile/tools.html#id4)

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
  - quant_kv: 存储`kv cache`时是否量化，可选为：`0, 1, 2, 3`，默认为`0`，含义如下：
    - 0: key和value都不量化
    - 1: 使用非对称8bit量化存储key
    - 2: 使用fp8格式寸处value
    - 3: 使用非对称8bit量化存储key，使用fp8格式寸处value
- 硬件配置
  - backend_type: 推理使用硬件后端类型，默认为：`"cpu"`
  - thread_num: 推理使用硬件线程数，默认为：`4`
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