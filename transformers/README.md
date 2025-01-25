# MNN-LLM
[中文版本](https://github.com/alibaba/MNN/wiki/llm)

An LLM inference engine developed based on MNN, supporting mainstream open-source LLM models. This functionality is divided into two parts:
- Model Export: Exports torch models to ONNX and then converts them to MNN models; exports tokenizer files, embedding files, etc.
- Model Inference: Supports inference of the exported models and enables text generation with LLM models.

## Model Export

`llmexport` is a tool for exporting LLM models, capable of exporting LLM models to ONNX and MNN formats.


### Usage
1. Clone the LLM project you want to export to your local environment, for example, Qwen2-0.5B-Instruct:
    ```sh
    git clone https://www.modelscope.cn/qwen/Qwen2-0.5B-Instruct.git
    ```
2. Run llmexport.py to export the model:

    ```sh
    cd ./transformers/llm/export
    # Export the model, tokenizer, embedding, and the corresponding MNN model
    python llmexport.py \
            --path /path/to/Qwen2-0.5B-Instruct \
            --export mnn
    ```
3. Exported Artifacts

The exported files include:

1. `config.json:` Configuration file for runtime, which can be manually modified.
2. `embeddings_bf16.bin:` Binary file containing the embedding weights, used during inference.
3. `llm.mnn:` The MNN model file, used during inference.
4. `llm.mnn.json:` JSON file corresponding to the MNN model, used for applying LoRA or GPTQ quantized weights.
5. `llm.mnn.weight:` MNN model weights, used during inference.
6. `llm.onnx:` ONNX model file without weights, not used during inference.
7. `llm_config.json:` Model configuration file, used during inference.

The directory structure is as follows:

```
.
└── model
     ├── config.json
     ├── embeddings_bf16.bin
     ├── llm.mnn
     ├── llm.mnn.json
     ├── llm.mnn.weight
     ├── onnx/
          ├──llm.onnx
           ├──llm.onnx.data
     ├── llm_config.json
     └── tokenizer.txt
```

### Features
+ Direct Conversion to MNN Model
Use `--export mnn` to directly convert to an MNN model. Note that you need to either install pymnn or specify the path to the MNNConvert tool using the `--mnnconvert` option. At least one of these conditions must be met. If pymnn is not installed and the MNNConvert tool's path is not specified via --mnnconvert, the llmexport.py script will search for the MNNConvert tool in the directory "../../../build/". Ensure that the MNNConvert file exists in this directory. This method currently supports exporting 4-bit and 8-bit models.

+ If you encounter issues with directly converting to an MNN model or require quantization with other bit depths (e.g., 5-bit/6-bit), you can first convert the model to an ONNX model using `--export onnx`. Then, use the MNNConvert tool to convert the ONNX model to an MNN model with the following command:

```
./MNNConvert --modelFile ../transformers/llm/export/model/onnx/llm.onnx --MNNModel llm.mnn --keepInputFormat --weightQuantBits=4 --weightQuantBlock=128 -f ONNX --transformerFuse=1 --allowCustomOp --saveExternalData
```

+ Supports dialogue testing with the model. Use `--test $query` to return the LLM's response.
+ Supports exporting with merged LoRA weights by specifying the directory of the LoRA weights using `--lora_path`.
+ Specify the quantization bit depth using `--quant_bit` and the block size for quantization using `--quant_block` .
+ Use `--lm_quant_bit` to specify the quantization bit depth for the lm_head layer's weights. If not specified, the bit depth defined by `--quant_bit` will be used.

### Parameters
```
usage: llmexport.py [-h] --path PATH [--type TYPE] [--lora_path LORA_PATH] [--dst_path DST_PATH] [--test TEST] [--export EXPORT]
                    [--quant_bit QUANT_BIT] [--quant_block QUANT_BLOCK] [--lm_quant_bit LM_QUANT_BIT]
                    [--mnnconvert MNNCONVERT]

llm_exporter

options:
  -h, --help            show this help message and exit
  --path PATH           path(`str` or `os.PathLike`):
                        Can be either:
                        	- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]
                        	- A path to a *directory* clone from repo like `../chatglm-6b`.
  --type TYPE           type(`str`, *optional*):
                        	The pretrain llm model type.
  --lora_path LORA_PATH
                        lora path, defaut is `None` mean not apply lora.
  --dst_path DST_PATH   export onnx/mnn model to path, defaut is `./model`.
  --test TEST           test model inference with query `TEST`.
  --export EXPORT       export model to an onnx/mnn model.
  --quant_bit QUANT_BIT
                        mnn quant bit, 4 or 8, default is 4.
  --quant_block QUANT_BLOCK
                        mnn quant block, default is 0 mean channle-wise.
  --lm_quant_bit LM_QUANT_BIT
                        mnn lm_head quant bit, 4 or 8, default is `quant_bit`.
  --mnnconvert MNNCONVERT
                        local mnnconvert path, if invalid, using pymnn.
```

## Model Inference

### Compilation

[Compile from Source](../compile/other.html#id4)
Add the required compilation macros during the standard compilation process:
```
-DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true
```

- To enable visual features, add the following macros:
```
-DLLM_SUPPORT_VISION=true -DMNN_BUILD_OPENCV=true -DMNN_IMGCODECS=true
```
- To enable audio features, add the following macros:
```
-DLLM_SUPPORT_AUDIO=true -DMNN_BUILD_AUDIO=true
```

#### mac / linux / windows
For macOS/Linux:
```
make build
cd build
cmake ../ -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true
make -j16
```

For x86 architecture, additionally include the `MNN_AVX512` macro:
```
make build
cd build
cmake ../ -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_AVX512=true
make -j16
```

#### Android：
Add the macros `MNN_ARM82` and `MNN_OPENCL`:
```
cd project/android
mkdir build_64
../build_64.sh "-DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_ARM82=true -DMNN_OPENCL=true -DMNN_USE_LOGCAT=true"
```

#### iOS: Refer to transformers/llm/engine/ios/README.md:
```
sh package_scripts/ios/buildiOS.sh "-DMNN_ARM82=true -DMNN_LOW_MEMORY=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_BUILD_LLM=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true"
```

#### Web
Refer to the environment setup at: https://mnn-docs.readthedocs.io/en/latest/compile/engine.html#web

- To compile the library and produce `libMNN.a`，`libMNN_Express.a`，`libllm.a`

```
mkdir buildweb
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-msimd128 -msse4.1" -DMNN_FORBID_MULTI_THREAD=ON -DMNN_USE_THREAD_POOL=OFF -DMNN_USE_SSE=ON -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true
make -j16
```

- Demo Compilation
To compile the demo:

```
emcc ../transformers/llm/engine/llm_demo.cpp -DCMAKE_CXX_FLAGS="-msimd128 -msse4.1" -I ../include -I ../transformers/llm/engine/include libMNN.a libllm.a express/libMNN_Express.a -o llm_demo.js --preload-file ~/qwen2.0_1.5b/ -s ALLOW_MEMORY_GROWTH=1 -o llm_demo.js
```

To test the compiled demo, use the following command:
```
node llm_demo.js ~/qwen2.0_1.5b/config.json ~/qwen2.0_1.5b/prompt.txt
```

### Usage
#### Runtime Configuration

##### Runtime Configuration
Place all the exported files required for model inference into the same folder. Add a `config.json` file to describe the model name and inference parameters. The directory structure should look as follows:
```
.
└── model_dir
     ├── config.json
     ├── embeddings_bf16.bin
     ├── llm_config.json
     ├── llm.mnn
     ├── llm.mnn.weight
     └── tokenizer.txt
```


##### Configuration Options
The configuration file supports the following options:
- Model File Information
  - `base_dir`: Directory where model files are loaded. Defaults to the directory of `config.json` or the model directory.
  - `llm_config`: Path to `llm_config.json`, resolved as `base_dir + llm_config`. Defaults to `base_dir + 'config.json'`.
  - `llm_model`: Path to `llm.mnn`, resolved as `base_dir + llm_model`. Defaults to `base_dir + 'llm.mnn`'.
  - `llm_weight`: Path to `llm.mnn.weight`, resolved as `base_dir + llm_weight`. Defaults to `base_dir + 'llm.mnn.weight'`
  - `block_model`: For segmented models, the path to `block_{idx}.mnn`, resolved as `base_dir + block_model`. Defaults to `base_dir + 'block_{idx}.mnn`.
  - `lm_model`: For segmented models, the path to `lm.mnn`, resolved as `base_dir + lm_model`. Defaults to `base_dir + 'lm.mnn'`.
  - `embedding_model`: If embedding uses a model, the path to the embedding is `base_dir + embedding_model`. Defaults to `base_dir + 'embedding.mnn'`.
  - `embedding_file`: If embedding uses a binary file, the path to the embedding is `base_dir + embedding_file`. Defaults to `base_dir + 'embeddings_bf16.bin'`.
  - `tokenizer_file`: Path to `tokenizer.txt`, resolved as `base_dir + tokenizer_file`. Defaults to `base_dir + 'tokenizer.txt'`.
  - `visual_model`:  If using a VL model, the path to the visual model is `base_dir + visual_model`. Defaults to `base_dir + 'visual.mnn'`.
- Inference Configuration
  - max_new_tokens: Maximum number of tokens to generate. Defaults to `512`
  - reuse_kv: Whether to reuse the `kv cache` in multi-turn dialogues. Defaults to `false`
  - quant_qkv: Whether to quantize query, key, value in the CPU attention operator. Options: `0`, `1`, `2`, `3`, `4`. Defaults to 0:
    - 0: Neither `key` nor `value` is quantized.
    - 1: Use asymmetric 8-bit quantization for `key`.
    - 2: Use `fp8` format to quantize value
    - 3: Use asymmetric `8-bit` quantization for key and `fp8` for `value`.
    - 4: Quantize both `key` and `value` while using asymmetric 8-bit quantization for `query` and `int8` matrix multiplication for `Q*K`.
  - use_mmap: Whether to use `mmap` to write weights to disk when memory is insufficient, avoiding overflow. Defaults to `false`. For mobile devices, it is recommended to set this to true.
  - kvcache_mmap: Whether to use `mmap` for KV Cache to write to disk when memory is insufficient, avoiding overflow. Defaults to `false`
  - tmp_path: Directory for disk caching when `mmap-related` features are enabled.
    - On iOS, a temporary directory can be created and set as follows:
    ```
    NSString *tempDirectory = NSTemporaryDirectory();llm->set_config("{\"tmp_path\":\"" + std::string([tempDirectory UTF8String]) + "\"}")
    ```

- Hardware Configuration
  - `backend_type`: Hardware backend type used for inference. Defaults to ：`"cpu"`
  - thread_num: Number of hardware threads used for CPU inference. Defaults to `4`. For OpenCL inference, use `68`.
  - precision: Precision strategy for inference. Defaults to `"low"`, preferring `fp16`.
  - memory: Memory strategy for inference. Defaults to `"low"`, enabling runtime quantization.

##### Config.json example
- `config.json`


  
- `llm_config.json`
  ```json
  {
      "hidden_size": 1536,
      "layer_nums": 28,
      "attention_mask": "float",
      "key_value_shape": [
          2,
          1,
          0,
          2,
          128
      ],
      "prompt_template": "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
      "is_visual": false,
      "is_single": true
  }
  ```

#### Inference Usage
The usage of `llm_demo` is as follows:
```
# Using config.json
## Interactive Chat
./llm_demo model_dir/config.json
## Replying to each line in the prompt
./llm_demo model_dir/config.json prompt.txt

# Without config.json, using default configuration
## Interactive Chat
./llm_demo model_dir/llm.mnn
## Replying to each line in the prompt
./llm_demo model_dir/llm.mnn prompt.txt

```


- For Visual Models
Embed image input in the prompt as follows:

```
<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Describe the content of the image.
```
Specify the image size:
```
<img><hw>280, 420</hw>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Describe the content of the image.

```
- For Audio Models
Embed audio input in the prompt as follows:

```
<audio>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav</audio>Describe the content of the audio.

```

#### GPTQ Weights
To use `GPTQ` weights, you can specify the path to the `Qwen2.5-0.5B-Instruct-GPTQ-Int4` model using the `--gptq_path PATH` option when exporting the `Qwen2.5-0.5B-Instruct` model. Use the following command:

```bash
# Export the GPTQ-quantized model
python llmexport.py --path /path/to/Qwen2.5-0.5B-Instruct --gptq_path /path/to/Qwen2.5-0.5B-Instruct-GPTQ-Int4 --export mnn

```

#### LoRA Weights
LoRA weights can be used in two ways:

Merge LoRA weights into the original model.
Export LoRA models separately.
The first approach is faster and simpler but does not support runtime switching of LoRA weights.
The second approach adds slight memory and computation overhead but is more flexible, supporting runtime switching of LoRA weights, making it suitable for multi-LoRA scenarios.

##### Merging LoRA

###### Export
To merge LoRA weights into the original model, specify the `--lora_path PATH` parameter during model export. By default, the model is exported with the merged weights. Use the following command:

```bash
# Export the model with merged LoRA weights
python llmexport.py --path /path/to/Qwen2.5-0.5B-Instruct --lora_path /path/to/lora --export mnn

```

###### Usage
Using the merged LoRA model is exactly the same as using the original model.

##### Separating LoRA

###### Export
To export LoRA as a separate model, supporting runtime switching, specify the --lora_path PATH parameter and include the --lora_split flag during model export. Use the following command:

```bash
python llmexport.py --path /path/to/Qwen2.5-0.5B-Instruct --lora_path /path/to/lora --lora_split --export mnn
```
After export, in addition to the original model files, a new `lora.mnn` file will be added to the folder, which is the LoRA model file.


###### Usage
- Using LoRA Model
  - Directly load the LoRA model by creating a `lora.json` configuration file, similar to running a merged LoRA model:

  ```json
  {
      "llm_model": "lora.mnn",
      "llm_weight": "base.mnn.weight",
  }
  ```
  - Runtime selection and switching between LoRA models:

  ```cpp
  // Create and load the base model
  std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
  llm->load();
  // Use the same object to selectively use multiple LoRA models, but cannot use them concurrently
  {
      // Add `lora_1` model on top of the base model; its index is `lora_1_idx`
      size_t lora_1_idx = llm->apply_lora("lora_1.mnn");
      llm->response("Hello lora1"); // Infer using `lora_1` model
      // Add `lora_2` model and use it
      size_t lora_2_idx = llm->apply_lora("lora_2.mnn");
      llm->response("Hello lora2"); // Infer using `lora_2` model
      // Select `lora_1` as the current model using its index
      llm->select_module(lora_1_idx);
      llm->response("Hello lora1"); // Infer using `lora_1` model
      // Release the loaded LoRA models
      llm->release_module(lora_1_idx);
      llm->release_module(lora_2_idx);
      // Select and use the base model
      llm->select_module(0);
      llm->response("Hello base"); // Infer using `base` model
  }
  // Use multiple objects to load and use multiple LoRA models concurrently
  {
      std::mutex creat_mutex;
      auto chat = [&](const std::string& lora_name) {
          MNN::BackendConfig bnConfig;
          auto newExe = Executor::newExecutor(MNN_FORWARD_CPU, bnConfig, 1);
          ExecutorScope scope(newExe);
          Llm* current_llm = nullptr;
          {
              std::lock_guard<std::mutex> guard(creat_mutex);
              current_llm = llm->create_lora(lora_name);
          }
          current_llm->response("Hello");
      };
      std::thread thread1(chat, "lora_1.mnn");
      std::thread thread2(chat, "lora_2.mnn");
      thread1.join();
      thread2.join();
  }
  ```