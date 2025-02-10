# MNNLLM iOS Application

[查看中文文档](./README-ZH.md)

## Introduction

This project is an iOS application based on the MNN engine, supporting local large-model multimodal conversations.

It operates fully offline with high privacy. Once the models are downloaded to the device, all conversations occur locally without any network uploads or processing.

## Features

1. **Model List**
   - Browse models supported by MNN.
   - Manage models: download and delete models.
   - Search for models locally.
   
2. **Multimodal Chat**
   - Text-to-text conversation.
   - Audio-to-text conversation.
   - Image-to-text conversation: capture images via camera or select from the gallery.

3. **Chat History**
   - View conversation history, with the ability to restore previous chat sessions.

### Application Preview:

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

## How to Build and Use

1. Clone the repository:

    ```shell
    git clone https://github.com/alibaba/MNN.git
    ```

2. Build the MNN.framework:

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

3. Copy the framework to the iOS project:

    ```shell
    mv MNN-iOS-CPU-GPU/Static/MNN.framework transformers/llm/engine/ios/MNN.framework
    project/ios/MNNLLMForiOS/MNN.framework
    ```

    Ensure the `Link Binary With Libraries` section includes the `MNN.framework`:
    
    ![framework](./assets/framework.png)

    If it's missing, add it manually:

    ![addFramework](./assets/addFramework.png)

    ![addFramework2](./assets/addFramework2.png)

4. Update iOS signing and build the project:

    ```shell
    cd project/ios/MNNLLMForiOS
    open MNNLLMiOS.xcodeproj
    ```

    In Xcode, go to `Signing & Capabilities > Team` and input your Apple ID and Bundle Identifier:

    ![signing](./assets/signing.png)

    Wait for the Swift Package to finish downloading before building.

## Notes

Due to memory limitations on iPhones, it is recommended to use models with 7B parameters or fewer to avoid memory-related crashes.

## References

- [Exyte/Chat](https://github.com/exyte/Chat)
- [stephencelis/CSQLite](https://github.com/stephencelis/SQLite.swift)
