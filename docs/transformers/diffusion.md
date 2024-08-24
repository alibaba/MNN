# 扩散模型

## 模型支持与下载

1. runwayml/stable-diffusion-v1-5
```
https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main
```
2. chilloutmix
```
https://modelscope.cn/models/wyj123456/chilloutmix
```
3. IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1
```
https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/tree/main
```
## 模型转换
### 将Huggingface的Stable Diffusion模型 转为onnx模型
```sh
cd mnn_path/transformers/diffusion/
python export/onnx_export.py \
    --model_path hf_sd_load_path \
    --output_path onnx_save_path
```
注意，上述脚本需要依赖torch/onnx/diffusers等库，可以安装conda环境：
```
conda env create -f env.yaml
conda activate ldm
```
在conda环境中执行模型转换脚本

### 将onnx模型转为mnn模型
新建diffusion mnn模型文件夹，将转好的mnn文件放在该文件夹下。
1. 实现encoder从onnx模型 -> mnn模型
```
./MNNConvert -f ONNX --modelFile onnx_save_path/text_encoder/model.onnx --MNNModel mnn_save_path/text_encoder.mnn --weightQuantBits 8 --bizCode biz
```
2. 实现denoiser unet从onnx模型 -> mnn模型
```
./MNNConvert -f ONNX --modelFile onnx_save_path/unet/model.onnx --MNNModel mnn_save_path/unet.mnn --transformerFuse --weightQuantBits 8 --bizCode biz
注意：对于非OpenCL后端推理，需要去掉--transformerFuse。
```
3. 实现decoder从onnx模型 -> mnn模型
```
./MNNConvert -f ONNX --modelFile onnx_save_path/vae_decoder/model.onnx --keepInputFormat --MNNModel mnn_save_path/vae_decoder.mnn --weightQuantBits 8 --bizCode biz
```
## 编译Diffusion Demo
### Linux/MAC/Windows上
```
cd mnn_path
mkdir build
cd build
cmake .. -DMNN_BUILD_DIFFUSION=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_OPENCL=ON -DMNN_SEP_BUILD=OFF -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
make -j32
```
### Android上
```
cd mnn_path/project/android/build
../build_64.sh -DMNN_BUILD_DIFFUSION=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_OPENCL=ON -DMNN_SEP_BUILD=OFF -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
../updateTest.sh
```
## 运行Diffusion Demo
```
./diffusion_demo <resource_path> <model_type> <output_image_name> <memory_mode> <backend_type> <input_text>
```
其中，resource_path 就是mnn模型文件的路径，除了mnn文件，还需要:
1. 将MNN目录transformers/diffusion/scheduler/alphas.txt文件拷贝到该文件夹下。
2. 针对stable-diffusion-v1-5/chilloutmix模型需要将huggingfacetokenizer目录下merges.txt和vocab.json拷贝到该文件夹中。
3. 针对Taiyi-Stable-Diffusion模型需要将huggingfacetokenizer目录下vocab.txt拷贝到该文件夹中。
4. model_type是目前支持的两种diffusion模型的类别。如果是stable-diffusion-v1-5/chilloutmix模型设为0，如果是Taiyi-Stable-Diffusion模型设为1。
5. output_image_name是生成图片的名字，默认图片位置在当前运行目录下。
6. memory_mode代表设备是否内存足够，设为0表示内存节约模式(demo中每个模型使用前等待初始化，用完释放)，1代表内存足够模式(所有模式启动时全初始化完，用时无需等待初始化)。
7. backend_type代表选择的运行后端。
8. input_text是文生图的prompt，如果是stable-diffusion-v1-5/chilloutmix模型建议英文prompt，如果是Taiyi-Stable-Diffusion建议中文prompt。

运行指令例如: 
```
./diffusion_demo mnn_sd1.5_path 0 demo.jpg 0 3 "a cute cat"
./diffusion_demo mnn_chilloutmix_path 0 demo.jpg 0 3 "a pure girl"
./diffusion_demo mnn_taiyi_path 1 demo.jpg 0 3 "一只可爱的猫"
```
## FAQ
1. Demo运行报错、段错误，怎么解决？
- 常见错误可能是设备内存不足，通常支持opencl fp16的设备需要保证3GB以上的内存，不支持fp16则需要6GB以上显存了。
2. 使用其他后端，出现报错，什么原因？
- 目前其他后端暂不支持transformer插件算子，需要在onnx->mnn模型转换阶段，去掉--transformerFuse。
