# llm-export

[English](./README_en.md)

llm-export是一个llm模型导出工具，能够将llm模型导出为onnx和mnn模型。

- 🚀 均完成`onnxruntime`正确性测试
- 🚀 优化原始代码，支持动态形状
- 🚀 优化原始代码，减少常量部分
- 🚀 使用[OnnxSlim](https://github.com/WeLoveAI/OnnxSlim)优化onnx模型，性能提升约5%; by [@inisis](https://github.com/inisis)
- 🚀 支持将lora权重导出为onnx和mnn

## 模型支持与下载
- [![Download][download-chatglm-6b-onnx]][release-chatglm-6b-onnx]
- [![Download][download-chatglm2-6b-onnx]][release-chatglm2-6b-onnx]
- [![Download][download-chatglm3-6b-onnx]][release-chatglm3-6b-onnx]
- [![Download][download-codegeex2-6b-onnx]][release-codegeex2-6b-onnx]
- [![Download][download-qwen-7b-chat-onnx]][release-qwen-7b-chat-onnx]
- [![Download][download-baichuan2-7b-chat-onnx]][release-baichuan2-7b-chat-onnx]
- [![Download][download-llama2-7b-chat-onnx]][release-llama2-7b-chat-onnx]
- [![Download][download-qwen-1.8b-chat-onnx]][release-qwen-1.8b-chat-onnx]
- [![Download][download-phi-2-onnx]][release-phi-2-onnx]
- [![Download][download-internlm-7b-onnx]][release-internlm-7b-onnx]
- [![Download][download-qwen-vl-onnx]][release-qwen-vl-onnx]
- [![Download][download-bge-large-zh-onnx]][release-bge-large-zh-onnx]
- [![Download][download-tinyllama-1.1b-chat-onnx]][release-tinyllama-1.1b-chat-onnx]
- [![Download][download-yi-6b-chat-onnx]][release-yi-6b-chat-onnx]
- [![Download][download-deepseek-7b-chat-onnx]][release-deepseek-7b-chat-onnx]
- [![Download][download-qwen1.5-0.5b-chat-onnx]][release-qwen1.5-0.5b-chat-onnx]
- [![Download][download-qwen1.5-1.8b-chat-onnx]][release-qwen1.5-1.8b-chat-onnx]
- [![Download][download-qwen1.5-4b-chat-onnx]][release-qwen1.5-4b-chat-onnx]
- [![Download][download-qwen1.5-7b-chat-onnx]][release-qwen1.5-7b-chat-onnx]
- [![Download][download-llama3-8b-instruct-onnx]][release-llama3-8b-instruct-onnx]

[download-chatglm-6b-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/chatglm-6b-onnx/total
[download-chatglm2-6b-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/chatglm2-6b-onnx/total
[download-chatglm3-6b-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/chatglm3-6b-onnx/total
[download-codegeex2-6b-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/codegeex2-6b-onnx/total
[download-qwen-7b-chat-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/qwen-7b-chat-onnx/total
[download-baichuan2-7b-chat-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/baichuan2-7b-chat-onnx/total
[download-llama2-7b-chat-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/llama2-7b-chat-onnx/total
[download-qwen-1.8b-chat-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/qwen-1.8b-onnx/total
[download-phi-2-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/phi-2-onnx/total
[download-internlm-7b-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/internlm-7b-onnx/total
[download-qwen-vl-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/qwen-vl-onnx/total
[download-bge-large-zh-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/bge-large-zh-onnx/total
[download-tinyllama-1.1b-chat-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/tinyllama-1.1b-chat-onnx/total
[download-yi-6b-chat-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/yi-6b-chat-onnx/total
[download-deepseek-7b-chat-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/deepseek-7b-chat-onnx/total
[download-qwen1.5-0.5b-chat-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/qwen1.5-0.5b-chat-onnx/total
[download-qwen1.5-1.8b-chat-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/qwen1.5-1.8b-chat-onnx/total
[download-qwen1.5-4b-chat-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/qwen1.5-4b-chat-onnx/total
[download-qwen1.5-7b-chat-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/qwen1.5-7b-chat-onnx/total
[download-llama3-8b-instruct-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/llama3-8b-instruct-onnx/total
[release-chatglm-6b-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/chatglm-6b-onnx
[release-chatglm2-6b-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/chatglm2-6b-onnx
[release-chatglm3-6b-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/chatglm3-6b-onnx
[release-codegeex2-6b-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/codegeex2-6b-onnx
[release-qwen-7b-chat-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/qwen-7b-chat-onnx
[release-baichuan2-7b-chat-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/baichuan2-7b-chat-onnx
[release-llama2-7b-chat-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/llama2-7b-chat-onnx
[release-qwen-1.8b-chat-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/qwen-1.8b-onnx
[release-phi-2-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/phi-2-onnx
[release-internlm-7b-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/internlm-7b-onnx
[release-qwen-vl-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/qwen-vl-onnx
[release-bge-large-zh-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/bge-large-zh-onnx
[release-tinyllama-1.1b-chat-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/tinyllama-1.1b-chat-onnx
[release-yi-6b-chat-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/yi-6b-chat-onnx
[release-deepseek-7b-chat-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/deepseek-7b-chat-onnx
[release-qwen1.5-0.5b-chat-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/qwen1.5-0.5b-chat-onnx
[release-qwen1.5-1.8b-chat-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/qwen1.5-1.8b-chat-onnx
[release-qwen1.5-4b-chat-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/qwen1.5-4b-chat-onnx
[release-qwen1.5-7b-chat-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/qwen1.5-7b-chat-onnx
[release-llama3-8b-instruct-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/llama3-8b-instruct-onnx

## 用法
1. 将该项目clone到本地
```sh
git clone git@github.com:wangzhaode/llm-export.git
```
2. 将需要导出的LLM项目clone到本地，如：chatglm2-6b
```sh
git clone https://huggingface.co/THUDM/chatglm2-6b
# 如果huggingface下载慢可以使用modelscope
git clone https://modelscope.cn/ZhipuAI/chatglm2-6b.git
```
3. 执行LLMExporter导出模型
```sh
cd mnn-llm
# 将chatglm2-6b分为embedding, blocks, lm分别导出为onnx并转换为mnn, 并导出tokenizer.txt
python llm_export.py \
        --path ../chatglm2-6b \
        --export_split \
        --export_token \
        --export_mnn \
        --onnx_path ./chatglm2-6b-onnx \
        --mnn_path  ./chatglm2-6b-mnn
```

## 功能
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

## 参数
```
usage: llm_export.py [-h] --path PATH
                     [--type {chatglm-6b,chatglm2-6b,chatglm3-6b,codegeex2-6b,Qwen-7B-Chat,Qwen-1_8B-Chat,Qwen-1_8B,Qwen-VL-Chat,Qwen1_5-0_5B-Chat,Qwen1_5-1_8B-Chat,Qwen1_5-4B-Chat,Qwen1_5-7B-Chat,Baichuan2-7B-Chat,Llama-2-7b-chat-ms,Llama-3-8B-Instruct,internlm-chat-7b,TinyLlama-1_1B-Chat,Yi-6B-Chat,deepseek-llm-7b-chat,phi-2,bge-large-zh,lora}]
                     [--lora_path LORA_PATH] [--onnx_path ONNX_PATH] [--mnn_path MNN_PATH] [--export_mnn] [--export_verbose] [--export_test] [--test TEST] [--export]
                     [--export_split] [--export_token] [--export_embed] [--export_visual] [--export_lm] [--export_block EXPORT_BLOCK] [--export_blocks] [--embed_bin]
                     [--embed_bf16] [--skip_slim]

llm_exporter

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           path(`str` or `os.PathLike`):
                        Can be either:
                        	- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]
                        	- A path to a *directory* clone from repo like `../chatglm-6b`.
  --type {chatglm-6b,chatglm2-6b,chatglm3-6b,codegeex2-6b,Qwen-7B-Chat,Qwen-1_8B-Chat,Qwen-1_8B,Qwen-VL-Chat,Qwen1_5-0_5B-Chat,Qwen1_5-1_8B-Chat,Qwen1_5-4B-Chat,Qwen1_5-7B-Chat,Baichuan2-7B-Chat,Llama-2-7b-chat-ms,Llama-3-8B-Instruct,internlm-chat-7b,TinyLlama-1_1B-Chat,Yi-6B-Chat,deepseek-llm-7b-chat,phi-2,bge-large-zh,lora}
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
