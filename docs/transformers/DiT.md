# DiT（Diffusion Transformer）模型

## 模型支持与下载

1. stable-diffusion-3.5-medium
```
https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/tree/main
```
## 模型转换
### 将DiT模型 转为onnx模型
```sh
optimum-cli export onnx \
    --model hf_model_path \
    --task stable-diffusion \
    --device cuda \
    onnx_save_path
```
注意，上述脚本需要依赖torch/onnx/diffusers等库，可以安装conda环境：
```
conda env create -f env.yaml
conda activate ldm
```
在conda环境中执行模型转换脚本

### 将onnx模型转为mnn模型
新建diffusion mnn模型文件夹 mnn_save_path ，将转好的mnn文件放在该文件夹下。

执行脚本
```
python3 convert_mnn_sd35.py onnx_save_path mnn_save_path "--weightQuantBits=8"
```

若希望在OpenCL / Metal后端上进一步加速，可加上--transformerFuse:
```
# 适用OpenCL / Metal后端推理
python3 convert_mnn_sd35.py onnx_save_path mnn_save_path "--weightQuantBits=8 --transformerFuse"
```

## 编译Diffusion Demo
### Linux/MAC/Windows上
```
cd mnn_path
mkdir build
cd build
cmake .. -DMNN_LOW_MEMORY=ON -DMNN_BUILD_DIFFUSION=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_OPENCL=ON -DMNN_SEP_BUILD=OFF -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DCMAKE_CXX_STANDARD=17
make -j32
```
### Android上
```
cd mnn_path/project/android/build
../build_64.sh -DMNN_LOW_MEMORY=ON -DMNN_BUILD_DIFFUSION=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_OPENCL=ON -DMNN_SEP_BUILD=OFF -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
../updateTest.sh
```
## 运行Diffusion Demo
```
./diffusion_sd35_demo <resource_path> <model_type> <memory_mode> <backend_type> <iteration_num> <random_seed> <output_image_name> <prompt_text>
```
其中，resource_path 就是mnn模型文件的路径，除了mnn文件，还需要:
### 资源拷贝
运行stable-diffusion-3.5-medium模型需要将huggingface模型中的tokenizer、tokenizer_2、tokenizer_3三个目录拷贝到resource_path文件夹中。
### 参数设置
```
1. model_type代表模型类型，0代表stable-diffusion-3.5-medium模型。
2. memory_mode代表设备是否内存足够，设为0表示内存节约模式(demo中每个模型使用前等待初始化，用完释放)，1代表内存足够模式(所有模式启动时全初始化完，性能快，运行时无需等待初始化), 2代表内存&性能折中模式(启动时初始化部分模型)。
3. backend_type代表选择的运行后端，如OpenCL/Metal/CPU等。
4. iteration_num代表文生图迭代次数，通常建议设置10到20之间。
5. random_seed代表固定输入噪声种子数，设置为负数表示随机生成噪声种子数。当随机噪声种子数生成图片质量不佳时，可以调节该参数种子数值。
```
### 提示词和图片名称设置
```
1. output_image_name是生成图片的名字，默认图片位置在当前运行目录下。
2. prompt_text是文生图的prompt，建议使用英文prompt。
```
### 运行命令示例
```
# 使用cuda后端运行stable-diffusion-3.5-medium模型
./diffusion_sd35_demo mnn_sd3.5_path 0 0 3 20 0 demo.jpg "A fluffy white kitten wearing a yellow oversized hoodie, holding a warm coffee cup, cozy rainy window background, Pixar style 3D render, cute, warm lighting, fuzzy texture."
```
## FAQ
1. Demo运行报错、段错误，怎么解决？
- 常见错误可能是设备内存不足，通常支持opencl fp16的设备需要保证10GB以上的内存，不支持fp16则需要20GB以上显存了。
2. 使用其他后端，出现报错，什么原因？
- 目前其他后端暂不支持transformer插件算子，需要在onnx->mnn模型转换阶段，去掉--transformerFuse。
