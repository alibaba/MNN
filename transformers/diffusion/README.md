# Diffusion使用方法

## 模型支持与下载

[Download-runwayml/stable-diffusion-v1-5]: 
https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main
[Download-IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1]:
https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/tree/main

## 模型转换
### 将Huggingface的Stable Diffusion模型 转为onnx模型
python export/onnx_export.py \
    --model_path hf_sd_load_path \
    --output_path onnx_save_path

### 将onnx模型转为mnn模型
新建diffusion mnn模型文件夹，将转好的mnn文件放在该文件夹下。
./MNNConvert -f ONNX --modelFile onnx_save_path/text_encoder/model.onnx --MNNModel mnn_save_path/text_encoder.mnn --weightQuantBits 8 --bizCode biz
./MNNConvert -f ONNX --modelFile onnx_save_path/unet/model.onnx --MNNModel mnn_save_path/unet.mnn --transformerFuse --weightQuantBits 8 --bizCode biz
./MNNConvert -f ONNX --modelFile onnx_save_path/vae_decoder/model.onnx --keepInputFormat --MNNModel mnn_save_path/vae_decoder.mnn --weightQuantBits 8 --bizCode biz

## 编译Diffusion Demo
### Linux/MAC/Windows上
cmake .. -DMNN_BUILD_DIFFUSION=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_OPENCL=ON -DMNN_SEP_BUILD=OFF -DMNN_SUPPORT_TRANSFORMER_FUSE=ON

### Android上
cd project/android/build
../build_64.sh -DMNN_BUILD_DIFFUSION=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_OPENCL=ON -DMNN_SEP_BUILD=OFF -DMNN_SUPPORT_TRANSFORMER_FUSE=ON

## 运行Diffusion Demo
./diffusion_demo <resource_path> <model_type> <output_image_name> <input_text>
其中，resource_path 就是mnn模型文件的路径，除了mnn文件，还需要
（1）将MNN目录transformers/diffusion/scheduler/alphas.txt文件拷贝到该文件夹下。
（2）针对stable-diffusion-v1-5模型需要将huggingfacetokenizer目录下merges.txt和vocab.json拷贝到该文件夹中。针对Taiyi-Stable-Diffusion模型需要将huggingfacetokenizer目录下vocab.txt拷贝到该文件夹中。

model_type是目前支持的两种diffusion模型的类别。如果是stable-diffusion-v1-5模型设为0，如果是Taiyi-Stable-Diffusion模型设为1。

output_image_name是生成图片的名字，默认图片位置在当前运行目录下。

input_text是文生图的prompt，如果是stable-diffusion-v1-5模型建议英文prompt，如果是Taiyi-Stable-Diffusion建议中文prompt。

运行指令例如: 
./diffusion_demo mnn_save_path 0 demo.jpg "a cute cat"
./diffusion_demo mnn_save_path 1 demo.jpg "一只可爱的猫"

