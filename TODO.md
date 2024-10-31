## Change Log
- [x] implement an independent `Sampler` Module.
- [x] implement 8 individual basic samplers: `greedy`, `temperature`, `topK`, `topP`, `minP`, `tfs`, `typical`, `penalty`. (can be configured through config.json)
- [x] implement `mixed` sampler, whose sampling order (Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature). one can change the samplers through configuring `mixed_samplers`. field in config.json
- [x] implement `PromptLib` to enable `chat_demo` for all LLM.
- [x] remove the `seq_len` control in `Llm` to `Sampler` and higher level modules to migrate design complexity. 
- [x] implement `Chat` to organize the workflow of a chatbot APP. 
- [x] change `#define FP16_QSCALE 0.25` in `CPUAttention` to ensure Llama3.2 FP16 correctness.
- [x] `chat_demo` tested on ubuntu 22.04, android(including ARM64, ARM82, OPENCL backend).  
- [ ] `chat_demo` supports visual model tasks (support Qwen2-VL demo). 
- [ ] `transformers/llm/engine/android` gives an text-only chatbot app based on `Qwen2.5-1.5B-Instruct`.
- [ ] `transformers/llm/engine/android` gives an text+image chatbot app based on `Qwen2-VL-2B-Instruct`.

Motivation: 
1. Sampler: performance, variety, different user-preferrence.
2. System Prompt: support history context, memory; few-shot generation (examples); role-play role profile.

## TODO Lists

### 0. Overall TODO Lists
- [ ] test ShareGPT, VQA, Audio...
- [ ] merge KV cache implementation
- [ ] verify the possibility of hetereogeneous computing (CPU + opencl/...)
- [ ] Kv cache + sampler

### 1. Engineering TODO Lists
- [x] llm-export convert Qwen2.5-1.5B-Instructx, Qwen2.5-3B-Instructx, Qwen2.5-7B-Instruct (Qwen2.5 language series) https://qwenlm.github.io/zh/blog/qwen2.5/ (<7B: 32K/8K, >=7B: 128K/8K)
- [x] llm-export convert Llama-3.2-1B-Instructx, Llama-3.2-3B-Instruct (Llama-3.2 language series) https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/ (128K)
- [ ] (optional) llm-export convert Llama-3.2-11B-Vision-Instruct (Llama-3.2 Vision series)
- [x] (optional) llm-export convert Qwen2-VL-2B-Instruct, Qwen2-VL-7B-Instruct (Qwen2-VL series)
- [ ] (optional) llm-export convert Qwen2-Audio-7B-Instruct (Qwen2 Audio series)
- [x] implement `Chat` application in transformers/llm/app/chat.
- [x] implement `LocalSampler` module. 
- [x] implement `PromptLib` module.
- [x] implement `Llm` module. (before StateCacheManager's existence)
- [x] build MNN on pc end.
- [x] build MNN and run on android (fp32+fp16)
- [ ] build and deploy an Llama-3.2-3B-Instruct model chatbot on Android cell phone, show the demo.
- [ ] add RAG implementation.

```bash
python llmexport.py --path ../../../model/Qwen2.5-1.5B-Instruct/ --dst_path ../../../model/qwen2_5-1_5b-instruct-mnn/ --export mnn --quant_bit 4 --quant_block 128

python llmexport.py --path ../../../model/Qwen2-VL-2B-Instruct/ --dst_path ../../../model/qwen2-vl-2b-instruct-mnn/ --export mnn --quant_bit 4 --quant_block 128

python llmexport.py --path ../../../model/Llama-3.2-3B-Instruct/ --dst_path ../../../model/llama3_2-3b-instruct-mnn --export mnn --quant_bit 4 --quant_block 128

adb push ../../model/qwen2_5-1_5b-instruct-mnn/ /data/local/tmp/llm
adb push ../../model/qwen2-vl-2b-instruct-mnn/ /data/local/tmp/llm
adb push ../../model/llama3_2-3b-instruct-mnn/ /data/local/tmp/llm

cd build/phone
adb push chat_demo libllm.so libMNN_CL.so libMNN_Express.so libMNN.so tools/cv/libMNNOpenCV.so /data/local/tmp/llm
```


### 2. Experiments TODO Lists
- [ ] test `Chat` on ShareGPT datasets, measuring time and space
- [ ] test VQA