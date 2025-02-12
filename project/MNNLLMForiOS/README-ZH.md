# MNNLLM iOS Application

## 介绍

本项目是一个基于MNN引擎，支持本地大模型多模态对话的iOS应用。

纯本地运行，隐私性强。当模型下载到本地之后，所有的对话都将在本地进行，不会有任何网络上传处理。


## 功能：

1. 模型列表
    - 获取 MNN 支持的模型礼拜；
    - 模型管理，支持下载和删除模型；
    - 模型搜索，支持本地模型搜索；
2. 多模态聊天对话
    - 文本到文本
    - 语音到文本
    - 图片到文本，图片可以拍摄输入或从图库中选择
3. 对话历史
    - 包含对话历史列表，可以还原对话场景


应用预览图：

<div style="display: flex; justify-content: center; align-items: center; text-align: center; width: 100%;">

<div style="flex: 0 0 20%; display: flex; flex-direction: column; align-items: center;">
<p style="margin: 0; font-weight: bold;">Text To Text</p>
<img alt="Icon" style="width: 80%;" src="./assets/text.PNG">
</div>

<div style="flex: 0 0 20%; display: flex; flex-direction: column; align-items: center;">
<p style="margin: 0; font-weight: bold;">Image To Text</p>
<img alt="Icon" style="width: 80%;" src="./assets/image.PNG">
</div>

<div style="flex: 0 0 20%; display: flex; flex-direction: column; align-items: center;">
<p style="margin: 0; font-weight: bold;">Audio To Text</p>
<img alt="Icon" style="width: 80%;" src="./assets/audio.PNG">
</div>

</div>

<div style="display: flex; justify-content: center; align-items: center; text-align: center; width: 100%;">

<div style="flex: 0 0 20%; display: flex; flex-direction: column; align-items: center;">
<p style="margin: 0; font-weight: bold;">Model List</p>
<img alt="Icon" style="width: 80%;" src="./assets/list.PNG">
</div>

<div style="flex: 0 0 20%; display: flex; flex-direction: column; align-items: center;">
<p style="margin: 0; font-weight: bold;">History</p>
<img alt="Icon" style="width: 80%;" src="./assets/history2.PNG">
</div>

<div style="flex: 0 0 20%; display: flex; flex-direction: column; align-items: center;">
<p style="margin: 0; font-weight: bold;">History</p>
<img alt="Icon" style="width: 80%;" src="./assets/history.PNG">
</div>

</div>


## 如何构建并使用

1. 下载仓库代码：

    ```shell
    git clone https://github.com/alibaba/MNN.git
    ```

2. 编译 MNN.framework:

    ```shell
    cd MNN/
    sh package_scripts/ios/buildiOS.sh "-DMNN_ARM82=true -DMNN_LOW_MEMORY=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_BUILD_LLM=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true
    -DMNN_METAL=ON
    -DMNN_BUILD_DIFFUSION=ON
    -DMNN_BUILD_OPENCV=ON
    -DMNN_IMGCODECS=ON
    -DMNN_OPENCL=OFF
    -DMNN_SEP_BUILD=OFF
    -DMNN_SUPPORT_TRANSFORMER_FUSE=ON"
    ```

3. 拷贝 framework 到 iOS 项目中

    ```shell
    mv MNN-iOS-CPU-GPU/Static/MNN.framework transformers/llm/engine/ios/MNN.framework
    
    project/ios/MNNLLMForiOS/MNN.framework

    ```

    确保 Link Binary With Libraried 中包含 MNN.framework
    
    ![](./assets/framework.png)

    如果没有包含，可以手动添加 MNN.framework:

    ![](./assets/addFramework.png)

    ![](./assets/addFramework2.png)

    

4. 修改 iOS 签名并编译项目
    
    ```shell
    cd project/ios/MNNLLMForiOS
    open MNNLLMiOS.xcodeproj
    ```

    在 Xcode 项目属性中 Signing & Capabilities > Team 输入自己的账号和Bundle Identifier
    
    ![signing](./assets/signing.png)


    等待 Swift Package 下载完成之后，进行编译使用。

## 注意

iPhone 因为内存有限，建议使用7B以及以下的模型，避免内存不足导致的崩溃。

## 引用

- [Exyte/Chat](https://github.com/exyte/Chat)
- [stephencelis/CSQLite](https://github.com/stephencelis/SQLite.swift)