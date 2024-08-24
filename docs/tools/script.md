# 脚本工具
一些功能性脚本，提供各种功能。

## apply_gptq.py
将GPTQ的权重写入到量化的MNN权重中。

### 用法
```
usage: apply_gptq.py [-h] --mnn_graph MNN_GRAPH --mnn_weight MNN_WEIGHT --gptq_tensor GPTQ_TENSOR

apply_gptq

options:
  -h, --help            show this help message and exit
  --mnn_graph MNN_GRAPH
                        mnn graph json path.
  --mnn_weight MNN_WEIGHT
                        mnn weight file path.
  --gptq_tensor GPTQ_TENSOR
                        gptq tensor path.
```

### 参数
- MNN_GRAPH: 模型计算图的json文件，获取方法：`./MNNDump2Json model.mnn model.json`
- MNN_WEIGHT:  模型的权重文件，如：`gptq.mnn.weight`
- GPTQ_TENSOR: GPTQ量化后的权重文件，`model.safetensor`

### 示例
使用该脚本生成gptq量化的权重`gptq.mnn.weight`
```sh
cd build
./MNNDump2Json model.mnn model.json
cp model.mnn.weight gptq.mnn.weight
python ../tools/script/apply_gptq.py --mnn_graph model.json --mnn_weight gptq.mnn.weight --gptq_tensor model.safetensor
```

## apply_lora.py

合并base模型的计算图和lora模型的权重文件，生成新的计算图。

### 用法
```sh
usage: apply_lora.py [-h] --base BASE --lora LORA [--scale SCALE] [--fuse FUSE] [--out OUT]

apply_lora

options:
  -h, --help     show this help message and exit
  --base BASE    base model json path.
  --lora LORA    lora dir path or *.safetensors path.
  --scale SCALE  lora scale: `alpha/r`.
  --fuse FUSE    fuse A and B.
  --out OUT      out file name.
```

### 参数
- BASE: base.json, base模型计算图的json文件，获取方法：`./MNNDump2Json base.mnn base.json`
- LORA: lora权重文件夹或者lora权重的safetensors
- SCALE: lora权重的scale, `lora_alpha / lora_r`, 一般为4.0
- FUSE: 是否将lora_A与lora_B合并成为一个lora权重，合并后模型较大
- OUT: 生成新的计算图文件名，默认为`lora.json`，转换为模型：`./MNNRevert2Buffer lora.json lora.mnn`

### 示例
使用该脚本生成lora对应的模型`lora.mnn`, 用法: [LoRA](../transformers/llm.html#lora)
```sh
cd build
./MNNDump2Json base.mnn base.json
python ../tools/script/apply_lora.py --base base.json --lora lora_dir
./MNNRevert2Buffer lora.json lora.mnn
```