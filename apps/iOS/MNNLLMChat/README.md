# MNNLLM iOS Application

[查看中文文档](./README-ZH.md)

## Introduction

This project is an iOS application based on the MNN engine, supporting local large-model multimodal conversations.

It operates fully offline with high privacy. Once the models are downloaded to the device, all conversations occur locally without any network uploads or processing.

[02/20/2025] Update:

- Support for configuring mmap and manual cache clearing

- Support for model downloads using the ModelScope source

| <img width="200" alt="image" src="./assets/usemmap_en.PNG" /> | <img width="200" alt="image" src="./assets/downloadSource_en.PNG" /> |

## Features

1. **Model List**
   - Browse models supported by MNN.
   - Manage models: download and delete models.
    - Support for switching between Hugging Face and ModelScope sources
   - Search for models locally.

2. **Multimodal Chat**: Supports full Markdown format output and configuration of mmap
   - Text-to-text conversation.
   - Audio-to-text conversation.
   - Image-to-text conversation: capture images via camera or select from the gallery.

3. **Chat History**
   - View conversation history, with the ability to restore previous chat sessions.


### Video Introduction

<img width="200" alt="image" src="./assets/introduction.gif" />


### Application Preview:

|  |  |  |
|--|--|--|
| **Text To Text**  | **Image To Text**  | **Audio To Text**  |
| ![Text To Text](./assets/text.PNG) | ![Image To Text](./assets/image.PNG) | ![Audio To Text](./assets/audio.jpg) |
| **Model List**  | **History**  | **History**  |
| ![Model List](./assets/list.PNG) | ![History](./assets/history2.PNG) | ![History](./assets/history.PNG) |

<p></p>

Additionally, the app supports edge-side usage of DeepSeek with Think mode:

<img src="./assets/deepseek.jpg" alt="deepThink" width="200" />



## How to Build and Use

1. Clone the repository:

    ```shell
    git clone https://github.com/alibaba/MNN.git
    ```

2. Build the MNN.framework:

    ```shell
    sh package_scripts/ios/buildiOS.sh "
    -DMNN_ARM82=ON
    -DMNN_LOW_MEMORY=ON
    -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
    -DMNN_BUILD_LLM=ON
    -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON
    -DMNN_METAL=ON
    -DMNN_BUILD_DIFFUSION=ON
    -DMNN_OPENCL=OFF
    -DMNN_SEP_BUILD=OFF
    -DLLM_SUPPORT_AUDIO=ON
    -DMNN_BUILD_AUDIO=ON
    -DLLM_SUPPORT_VISION=ON
    -DMNN_BUILD_OPENCV=ON
    -DMNN_IMGCODECS=ON
    "
    ```

3. Copy the framework to the iOS project:

    ```shell
    mv MNN-iOS-CPU-GPU/Static/MNN.framework apps/iOS/MNNLLMChat/MNN.framework
    ```

    Ensure the `Link Binary With Libraries` section includes the `MNN.framework`:

    <img src="./assets/framework.png" alt="deepThink" width="400" />

    If it's missing, add it manually:

    <img src="./assets/addFramework.png" alt="deepThink" width="200" />
    <img src="./assets/addFramework2.png" alt="deepThink" width="200" />


4. Update iOS signing and build the project:

    ```shell
    cd apps/iOS/MNNLLMChat
    open MNNLLMiOS.xcodeproj
    ```

    In Xcode, go to `Signing & Capabilities > Team` and input your Apple ID and Bundle Identifier:

    ![signing](./assets/signing.png)

    Wait for the Swift Package to finish downloading before building.

## Notes

Due to memory limitations on iPhones, it is recommended to use models with 7B parameters or fewer to avoid memory-related crashes.

Here is the professional technical translation of the provided text:

---

## Local Debugging

If we want to directly download the models to the computer for debugging without downloading them through the app, we can follow these steps:

1. First, download the MNN-related models from [Hugging Face](https://huggingface.co/taobao-mnn) or [Modelscope](https://modelscope.cn/organization/MNN):


    <img width="400" alt="image" src="./assets/copyLocalModel.png" />


2. After downloading, drag all the files from the model folder into the project's LocalModel folder:

    <img width="300" alt="image" src="./assets/copyLocalModel2.png" />

3. Ensure that the above files are included in the Copy Bundle Resources section:

    <img width="400" alt="image" src="./assets/copyLocalMode3.png" />

4. Comment out the model download code:

    ```Swift
    /*
    try await modelClient.downloadModel(model: model) { progress in
        Task { @MainActor in
            DispatchQueue.main.async {
                self.downloadProgress[model.modelId] = progress
            }
        }
    }
    */
    ```


5. Modify the Model Loading Method

    Modify the `LLMInferenceEngineWrapper` class:

    ```Swift
    // BOOL success = [self loadModelFromPath:modelPath];
    // MARK: Test Local Model
    BOOL success = [self loadModel];
    ```


6. Run the project, navigate to the chat page, and perform model interactions and debugging.


## References

- [Exyte/Chat](https://github.com/exyte/Chat)
- [stephencelis/CSQLite](https://github.com/stephencelis/SQLite.swift)
- [swift-transformers](https://github.com/huggingface/swift-transformers/)