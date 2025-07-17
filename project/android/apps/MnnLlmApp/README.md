# MNN-LLM Android App
**The App has been renamed "MNN Chat" and the home page has been moved to [Here](../../../../apps/Android/MnnLlmChat/README.md)**

[中文版本](./README_CN.md)
## Introduction
This is our full multimodal language model (LLM) Android app

<p align="center">
  <img width="20%" alt="Icon"  src="../../../../apps/Android/MnnLlmChat/assets/image_home.jpg" style="margin: 0 10px;">
  <img width="20%" alt="Icon" src="../../../../apps/Android/MnnLlmChat/assets/image_diffusion.jpg" style="margin: 0 10px;">
  <img width="20%" alt="Icon" src="../../../../apps/Android/MnnLlmChat/assets/image_sound.jpg" style="margin: 0 10px;">
  <img width="20%" alt="Icon" src="../../../../apps/Android/MnnLlmChat/assets/image_image.jpg" style="margin: 0 10px;">
</p>


### Features

+ **Multimodal Support:** Enables functionality across diverse tasks, including text-to-text, image-to-text, audio-to-text, and text-to-image generation (via diffusion models).

+ **CPU Inference Optimization:** MNN-LLM demonstrates exceptional performance in CPU benchmarking in Android, achieving prefill speed improvements of 8.6x over llama.cpp and 20.5x over fastllm, with decoding speeds that are 2.3x and 8.9x faster, respectively. the following is a comparison between llama.cpp and MNN-LLM on Android inferencing qwen-7b.
<p align="center">
  <img width="60%"   src="./assets/compare.gif" style="margin: 0 10px;">
</p>

+ **Broad Model Compatibility:** Supports multiple leading model providers, such as Qwen, Gemma, Llama (including TinyLlama and MobileLLM), Baichuan, Yi, DeepSeek, InternLM, Phi, ReaderLM, and Smolm.

+ **Privacy First:** Runs entirely on-device, ensuring complete data privacy with no information uploaded to external servers.


# How to Use
+ you can download the app from [Releases](#releases) or [build it yourself](#development);
+ After installing the application, you can browse all supported models, download them, and interact with them directly within the app.;
+ Additionally, you can access your chat history in the sidebar and revisit previous conversations seamlessly.

 !!!warning!!! This version has been tested exclusively on the OnePlus 13 and Xiaomi 14 Ultra, Due to the demanding performance requirements of large language models (LLMs), many budget or low-spec devices may experience issues such as slow inference speeds, application instability, or even failure to run entirely. and its stability on other devices cannot be guaranteed. If you encounter any issues, please feel free to open an issue for assistance.


# Releases
## Version 0.2.2
+ Click here to [download](https://meta.alicdn.com/data/mnn/mnn_chat_d_0_2_2.apk)
+ Support mmap for speed up laoding speed.
+ Add version update checker

## Version 0.2.1
+ Click here to [download](https://meta.alicdn.com/data/mnn/mnn_chat_d_0_2_1_1.apk)
+ Support for ModelScope downloads
+ Optimization of DeepSeek's multi-turn conversation capabilities and UI presentation
+ Added support for including debug information when submitting feedback or issues
<p align="center">
  <img width="20%" alt="Icon"  src="./assets/deepseek_support.gif" style="margin: 0 10px;">
</p>

## Version 0.2
+ Click here to [download](https://meta.alicdn.com/data/mnn/mnn_llm_app_debug_0_2_0.apk)
+ Optimized for DeepSeek R1 1.5B
+ Added support for Markdown
+ Resolved several bugs and improved stability

## Version 0.1
+ Click here to [download](https://meta.alicdn.com/data/mnn/mnn_llm_app_debug_0_1.apk)
+ this is our first public released version; you can :
  + search all our supported models, download  and chat with it in the app; 
  + diffusion model:
    + stable-diffusion-v1-5
  + audio model:
    + qwen2-audio-7b
  + visual models:
    + qwen-vl-chat
    + qwen2-vl-2b
    + qwen2-vl-7b




# About MNN-LLM
MNN-LLM is a versatile inference framework designed to optimize and accelerate the deployment of large language models on both mobile devices and local PCs, addressing challenges like high memory consumption and computational costs through innovations such as model quantization, hybrid storage, and hardware-specific optimizations. In CPU benchmarking, MNN-LLM excels, achieving prefill speed boosts of 8.6x over llama.cpp and 20.5x over fastllm, complemented by decoding speeds that are 2.3x and 8.9x faster, respectively. In
GPU-based assessments, MNN-LLM’s performance slightly declines
compared to MLC-LLM, particularly when using Qwen2-7B with shorter prompts, due to MLC-LLM’s advantageous symmetric quantization technique. MNN-LLM excels, achieving up to 25.3x faster prefill and 7.1x faster decoding than llama.cpp, and 2.8x and 1.7x improvements over MLC-LLM, respectively.
 For more detailed information, please refer to the paper:[MNN-LLM: A Generic Inference Engine for Fast Large LanguageModel Deployment on Mobile Devices](https://dl.acm.org/doi/pdf/10.1145/3700410.3702126) 


# Acknowledgements
This project is built upon the following open-source projects:

+ [progress-dialog](https://github.com/techinessoverloaded/progress-dialog)
+ [okhttp](https://github.com/square/okhttp)
+ [retrofit](https://github.com/square/retrofit)
+ [Android-SpinKit](https://github.com/ybq/Android-SpinKit)
+ [expandable-fab](https://github.com/nambicompany/expandable-fab)
+ [Android-Wave-Recorder](https://github.com/squti/Android-Wave-Recorder)