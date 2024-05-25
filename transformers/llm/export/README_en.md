# llm-export

[中文](./README_en.md)

llm-export is a tool for exporting llm models, capable of converting llm models into ONNX or MNN models.
- 🚀 All passed `onnxruntime` correctness tests
- 🚀 Optimized the original code to support dynamic shapes
- 🚀 Optimized the original code to reduce the constant portion
- 🚀 Using [OnnxSlim](https://github.com/WeLoveAI/OnnxSlim) slim onnx model，speed up 5%; by [@inisis](https://github.com/inisis)
- 🚀 Support export lora weight to onnx or MNN model

## Model Support and Downloads

## Usage
1. Clone this project locally
```sh
git clnoe git@github.com:wangzhaode/llm-export.git
```
2. Clone the LLM project that you want to export locally, such as: chatglm2-6b
```sh
git clone https://huggingface.co/THUDM/chatglm2-6b
# If downloading from Hugging Face is slow, you can use ModelScope
git clone https://modelscope.cn/ZhipuAI/chatglm2-6b.git
```
3. Execute LLMExporter to export the model
```sh
cd mnn-llm
# Divide chatglm2-6b into embedding, blocks, lm, export each as ONNX and convert to MNN, and also export tokenizer.txt
python llm_export.py \
        --path ../chatglm2-6b \
        --export_split \
        --export_token \
        --export_mnn \
        --onnx_path ./chatglm2-6b-onnx \
        --mnn_path  ./chatglm2-6b-mnn 
```

## Features
- Supports exporting the entire model as a single ONNX model, use --export
- Supports exporting the model in segments as multiple models, use --export_split
- Supports exporting the model's vocabulary to a text file, each line representing a token; tokens are encoded using base64, use --export_verbose
- Supports exporting the model's Embedding layer as an ONNX model, use --export_embed, also supports bf16 format, use --embed_bf16
- Supports layered export of the model's blocks, use --export_blocks to export all layers; use --export_block $id to export a specified layer
- Supports exporting the model's lm_head layer as an ONNX model, use --export_lm
- Supports exporting the VL model's visual model as an ONNX model, use --export_visual
- Supports conducting a dialogue test on the model, using --test $query will return the llm's response
- Supports verifying the consistency of results using onnxruntime after exporting the ONNX model, use --export_test
- Supports exporting the tokenizer as a text file, use --export_token
- Supports converting the exported ONNX model to an MNN model, with default conversion to non-symmetric 4bit quantization, use --export_mnn
- Specify export paths using --onnx_path and --mnn_path
- Default using onnx-slim, skip using --skip_slim

## Commad Args
```
usage: llm_export.py [-h] --path PATH
                     [--type {chatglm-6b,chatglm2-6b,chatglm3-6b,codegeex2-6b,Qwen-7B-Chat,Qwen-1_8B-Chat,Qwen-VL-Chat,Baichuan2-7B-Chat,Llama-2-7b-chat-ms,internlm-chat-7b,TinyLlama-1_1B-Chat,Yi-6B-Chat,deepseek-llm-7b-chat,phi-2,bge-large-zh}]
                     [--onnx_path ONNX_PATH] [--mnn_path MNN_PATH] [--export_mnn] [--export_verbose] [--export_test] [--test TEST] [--export] [--export_split] [--export_token] [--export_embed] [--export_visual] [--export_lm]
                     [--export_block EXPORT_BLOCK] [--export_blocks] [--embed_bf16] [--skip_slim]

llm_exporter

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           path(`str` or `os.PathLike`):
                        Can be either:
                                - A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]
                                - A path to a *directory* clone from repo like `../chatglm-6b`.
  --type {chatglm-6b,chatglm2-6b,chatglm3-6b,codegeex2-6b,Qwen-7B-Chat,Qwen-1_8B-Chat,Qwen-VL-Chat,Baichuan2-7B-Chat,Llama-2-7b-chat-ms,internlm-chat-7b,TinyLlama-1_1B-Chat,Yi-6B-Chat,deepseek-llm-7b-chat,phi-2,bge-large-zh}
                        type(`str`, *optional*):
                                The pretrain llm model type.
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
  --embed_bf16          using `bfloat16` replace `float32` in embedding.
  --skip_slim           Whether or not to skip onnx-slim.
```
