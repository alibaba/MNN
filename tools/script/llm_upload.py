# llm_upload.py
# Author: Zhaode Wang
# Time: 2024-12-31 10:41
# Description: llm models upload script

import os
import argparse

class LlmUploader:
    def __init__(self, args):
        self.model_dir = args.path
        self.hf_token = args.hf_token
        self.ms_token = args.ms_token
        self.me_token = args.me_token
        self.bits = args.bits
        self.name = args.name
        self.src = args.src
        if self.name is None:
            self.name = os.path.basename(self.model_dir)
        if self.src is None:
            self.src = self.name[:-4]

    def build_gitattributes(self):
        gitattributes = """*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.bin.* filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.lfs.* filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rar filter=lfs diff=lfs merge=lfs -text
saved_model/**/* filter=lfs diff=lfs merge=lfs -text
*.tar.* filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.xz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.zstandard filter=lfs diff=lfs merge=lfs -text
*.tfevents* filter=lfs diff=lfs merge=lfs -text
*.db* filter=lfs diff=lfs merge=lfs -text
*.ark* filter=lfs diff=lfs merge=lfs -text
**/*ckpt*data* filter=lfs diff=lfs merge=lfs -text
**/*ckpt*.meta filter=lfs diff=lfs merge=lfs -text
**/*ckpt*.index filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.mnn filter=lfs diff=lfs merge=lfs -text
*.mnn.* filter=lfs diff=lfs merge=lfs -text
*.weight filter=lfs diff=lfs merge=lfs -text
    """
        with open(f'{self.model_dir}/.gitattributes', 'w') as f:
            f.write(gitattributes)

    def build_ms_configuration(self):
        configuration = '{"framework":"other","task":"text-generation"}'
        with open(f'{self.model_dir}/configuration.json', 'w') as f:
            f.write(configuration)

    def build_ms_readme(self):
        readme = f"""---
frameworks:
- MNN
license: Apache License 2.0
tasks:
- text-generation

model-type:
- transformer

domain:
- nlp

language:
- cn

tags:
- instruction-tuned

tools:
- MNN
---

# {self.name}

## 介绍（Introduction）
此模型是使用[llmexport](https://github.com/alibaba/MNN/tree/master/transformers/llm/export)从{self.src}导出的{self.bits}bit量化版本的MNN模型。

## 下载
```bash
#安装ModelScope
pip install modelscope
```
```bash
#命令行工具下载
modelscope download --model 'MNN/{self.name}' --local_dir 'path/to/dir'
```
```python
#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('MNN/{self.name}')
```
Git下载
```bash
#Git模型下载
git clone https://www.modelscope.cn/MNN/{self.name}.git
```

## 使用
```bash
# 下载MNN源码
git clone https://github.com/alibaba/MNN.git

# 编译
cd MNN
mkdir build && cd build
cmake .. -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true
make -j

# 运行
./llm_demo /path/to/{self.name}/config.json prompt.txt
```

## 文档
[MNN-LLM](https://mnn-docs.readthedocs.io/en/latest/transformers/llm.html#)
    """
        readme_file = os.path.join(self.model_dir, 'README.md')
        if os.path.exists(readme_file):
            os.remove(readme_file)
        with open(readme_file, 'w') as f:
            f.write(readme)

    def build_hf_readme(self):
        readme = f"""---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- chat
---
# {self.name}

## Introduction
This model is a {self.bits}-bit quantized version of the MNN model exported from {self.src} using [llmexport](https://github.com/alibaba/MNN/tree/master/transformers/llm/export).

## Download
```bash
# install huggingface
pip install huggingface
```
```bash
# shell download
huggingface download --model 'taobao-mnn/{self.name}' --local_dir 'path/to/dir'
```
```python
# SDK download
from huggingface_hub import snapshot_download
model_dir = snapshot_download('taobao-mnn/{self.name}')
```

```bash
# git clone
git clone https://www.modelscope.cn/taobao-mnn/{self.name}
```

## Usage
```bash
# clone MNN source
git clone https://github.com/alibaba/MNN.git

# compile
cd MNN
mkdir build && cd build
cmake .. -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true
make -j

# run
./llm_demo /path/to/{self.name}/config.json prompt.txt
```

## Document
[MNN-LLM](https://mnn-docs.readthedocs.io/en/latest/transformers/llm.html#)
    """
        configuration_file = os.path.join(self.model_dir, 'configuration.json')
        if os.path.exists(configuration_file):
            os.remove(configuration_file)
        readme_file = os.path.join(self.model_dir, 'README.md')
        if os.path.exists(readme_file):
            os.remove(readme_file)
        with open(readme_file, 'w') as f:
            f.write(readme)

    def build_me_readme(self):
        readme = f"""---
license: apache-2.0
---

# {self.name}

## 介绍（Introduction）
此模型是使用[llmexport](https://github.com/alibaba/MNN/tree/master/transformers/llm/export)从{self.src}导出的{self.bits}bit量化版本的MNN模型。

## 下载
```bash
#安装openmind_hub
pip install openmind_hub
```
```python
#SDK模型下载
from openmind_hub import snapshot_download
snapshot_download(repo_id="MNN/{self.name}", token="your_token", repo_type="model")
```
Git下载
```bash
git lfs install
#Git模型下载
git clone https://modelers.cn/MNN/{self.name}.git
```

## 使用
```bash
# 下载MNN源码
git clone https://github.com/alibaba/MNN.git

# 编译
cd MNN
mkdir build && cd build
cmake .. -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true
make -j

# 运行
./llm_demo /path/to/{self.name}/config.json prompt.txt
```

## 文档
[MNN-LLM](https://mnn-docs.readthedocs.io/en/latest/transformers/llm.html#)
    """
        readme_file = os.path.join(self.model_dir, 'README.md')
        if os.path.exists(readme_file):
            os.remove(readme_file)
        with open(readme_file, 'w') as f:
            f.write(readme)

    def upload_modelscope(self):
        self.build_ms_configuration()
        self.build_ms_readme()
        model_id = f'MNN/{self.name}'
        from modelscope.hub.api import HubApi
        api = HubApi()
        api.login(self.ms_token)
        api.push_model(
            model_id=model_id,
            model_dir=self.model_dir,
            lfs_suffix=['*.mnn', '*.weight', "*.bin"]
        )

    def upload_huggingface(self):
        self.build_hf_readme()
        model_id = f'taobao-mnn/{self.name}'
        from huggingface_hub import HfApi
        from huggingface_hub import login
        login(token=self.hf_token)
        api = HfApi()
        try:
            api.create_repo(repo_id=model_id)
        except:
            pass
        api.upload_folder(
            folder_path=self.model_dir,
            repo_id=model_id,
            repo_type="model"
        )

    def upload_modelers(self):
        os.environ['HUB_WHITE_LIST_PATHS'] = '/'
        self.build_me_readme()
        model_id = f'MNN/{self.name}'
        import openmind_hub
        from openmind_hub import OmApi
        api = OmApi(token=self.me_token)
        try:
            api.create_repo(repo_id=model_id)
        except:
            pass

        # openmind_hub.upload_folder(
        api.upload_folder(
            folder_path=self.model_dir,
            repo_id=model_id,
            repo_type="model",
            token=self.me_token,
        )

    def upload(self):
        self.build_gitattributes()
        if self.hf_token is not None:
            self.upload_huggingface()
        if self.ms_token is not None:
            self.upload_modelscope()
        if self.me_token is not None:
            self.upload_modelers()

def main():
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--path', type=str, required=True, help='uploda model path')
    parser.add_argument('--hf_token', type=str, default=None, help='huggingface token.')
    parser.add_argument('--ms_token', type=str, default=None, help='modelscope token.')
    parser.add_argument('--me_token', type=str, default=None, help='modelers token.')
    parser.add_argument('--bits', type=int, default=4, help='quant bits, default is `4`.')
    parser.add_argument('--name', type=str, default=None, help='model name, default is path dir name.')
    parser.add_argument('--src', type=str, default=None, help='model src, default is model_name[:-4].')

    args = parser.parse_args()
    uploader = LlmUploader(args)
    uploader.upload()

if __name__ == '__main__':
    main()
