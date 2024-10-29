## TODO Lists

### 1. Engineering TODO Lists
- [x] llm-export convert Qwen2.5-1.5B-Instructx, Qwen2.5-3B-Instructx, Qwen2.5-7B-Instruct (Qwen2.5 language series) https://qwenlm.github.io/zh/blog/qwen2.5/ (<7B: 32K/8K, >=7B: 128K/8K)
- [x] llm-export convert Llama-3.2-1B-Instructx, Llama-3.2-3B-Instruct (Llama-3.2 language series) https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/ (128K)
- [ ] (optional) llm-export convert Llama-3.2-11B-Vision-Instruct (Llama-3.2 Vision series)
- [ ] (optional) llm-export convert Qwen2-VL-2B-Instruct, Qwen2-VL-7B-Instruct (Qwen2-VL series)
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
```


### 2. Experiments TODO Lists
- [ ] test `Chat` on ShareGPT datasets, measuring time and space.