# llm-export

[English](./README_en.md)

llm-export是一个llm模型导出工具，能够将llm模型导出为onnx和mnn模型。

- 🚀 优化原始代码，支持动态形状
- 🚀 优化原始代码，减少常量部分
- 🚀 使用[OnnxSlim](https://github.com/inisis/OnnxSlim)优化onnx模型，性能提升约5%; by [@inisis](https://github.com/inisis)
- 🚀 支持将lora权重导出为onnx和mnn
- 🚀 Onnx推理代码[OnnxLLM](https://github.com/inisis/OnnxLLM)

## 安装
```sh
# pip install
pip install llmexport

# git install
pip install git+https://github.com/wangzhaode/llm-export@master

# local install
git clone https://github.com/wangzhaode/llm-export && cd llm-export/
pip install .
```

## 用法

1. 下载模型
```sh
git clone https://huggingface.co/Qwen/Qwen2-1.5B-Instruct
# 如果huggingface下载慢可以使用modelscope
git clone https://modelscope.cn/qwen/Qwen2-1.5B-Instruct.git
```
2. 测试模型
```sh
# 测试文本输入
llmexport --path Qwen2-1.5B-Instruct --test "你好"
# 测试图像文本
llmexport --path Qwen2-VL-2B-Instruct  --test "<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>介绍一下图片里的内容"
```

3. 导出模型
```sh
# 将Qwen2-1.5B-Instruct导出为onnx模型
llmexport --path Qwen2-1.5B-Instruct --export onnx
# 将Qwen2-1.5B-Instruct导出为mnn模型, 量化参数为4bit, blokc-wise = 128
llmexport --path Qwen2-1.5B-Instruct --export mnn --quant_bit 4 --quant_block 128
```

## 功能
- 支持将模型为onnx或mnn模型，使用`--export onnx`或`--export mnn`
- 支持对模型进行对话测试，使用`--test $query`会返回llm的回复内容
- 默认会使用onnx-slim对onnx模型进行优化，跳过该步骤使用`--skip_slim`
- 支持合并lora权重后导出，指定lora权重的目录使用`--lora_path`
- 制定量化bit数使用`--quant_bit`；量化的block大小使用`--quant_block`
- 使用`--lm_quant_bit`来制定lm_head层权重的量化bit数，不指定则使用`--quant_bit`的量化bit数
- 支持使用自己编译的`MNNConvert`，使用`--mnnconvert`

## 参数
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
  --onnx_slim           Whether or not to use onnx-slim.
  --quant_bit QUANT_BIT
                        mnn quant bit, 4 or 8, default is 4.
  --quant_block QUANT_BLOCK
                        mnn quant block, default is 0 mean channle-wise.
  --lm_quant_bit LM_QUANT_BIT
                        mnn lm_head quant bit, 4 or 8, default is `quant_bit`.
  --mnnconvert MNNCONVERT
                        local mnnconvert path, if invalid, using pymnn.
```

## 支持模型

- llama/llama2/llama3/llama3.2/tinyllama
- qwen/qwen1.5/qwen2/qwen-vl/qwen2-vl/qwen2.5
- baichuan2/phi-2/internlm/yi/deepseek
- chatglm/codegeex/chatglm2/chatglm3
- phi-2/gemma-2
