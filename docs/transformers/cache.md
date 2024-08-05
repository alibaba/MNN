## StateCacheManager

### Compilation

LLM and TRANSFORMER_FUSE shall be set ON: `-DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON`

```bash
cmake .. -DCMAKE_CXX_STANDARD=17 -DMNN_USE_SYSTEM_LIB=OFF -DMNN_BUILD_SHARED_LIBS=ON -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON -DMNN_BUILD_CONVERTER=ON -DMNN_PORTABLE_BUILD=ON -DTFMODEL_OPTIMIZE=ON -DMNN_LOW_MEMORY=ON -DMNN_AVX512=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_TEST=ON
make -j20

./llm_demo /home/hzx/Desktop/ANL/Project/LLM/MNN-LLM/model/qwen1_5-4b-chat-mnn-f/llm_config.json
```

Transformer path: 
Download from https://github.com/wangzhaode/mnn-llm/releases
`/home/hzx/Desktop/ANL/Project/LLM/MNN-LLM/model/qwen1_5-4b-chat-mnn-f/`
`./llm_demo /home/hzx/Desktop/ANL/Project/LLM/MNN-LLM/model/qwen1_5-4b-chat-mnn-f/llm_config.json`

export
```bash
cd transformers/llm/export/
python llm_export.py \
        --path ../../../model/Qwen1_5-4B-Chat \
        --type Qwen1_5-4B-Chat \
        --export_split \
        --export_token \
        --export_mnn \
        --export_embed --embed_bin --embed_bf16 \
        --onnx_path ../../../model/qwen1_5-4b-chat-onnx2 \
        --mnn_path  ../../../model/qwen1_5-4b-chat-mnn2
```

Currently fuse attention haven't added to python MNN package converter. To use it, you need directly use MNNConvert.
```bash
for i in $(seq 0 39)
do
    ./build/MNNConvert -f ONNX --modelFile ./model/qwen1_5-4b-chat-onnx/block_${i}.onnx --MNNModel ./model/qwen1_5-4b-chat-mnn-ff/block_${i}.mnn --weightQuantBits 4 --weightQuantAsymmetric --transformerFuse
done
```


### Files

`core/StateCacheManager.hpp` and `core/StateCacheManager.cpp`

### Config

1. StateCacheType: implementation type of StateCacheManager
2. StateCacheQuantType: quantization type of StateCacheManager

2 fields are added to RuntimeHint, can be modified by `runtime_manager_->setHint(mode, value);`.
```cpp
struct RuntimeHint {
    // 0: Defer, 1: Eager
    int memoryAllocatorType = 0;
    int winogradMemoryUsed = 3;
    
    // 0-100, 50 means litter core has 50% capacity of large core
    int cpuDecreaseRate = 50;
    int dynamicQuantOption = 0;

    // 0: Do not quantize kvcache, just store float
    // 1: Only quantize key cache, use int8 asymmetric quantization 
    // 2: Only quantize value cache, use fp8 quantization
    // 3: quantize both key and value cache as described above
    int kvcacheQuantOption = (int)MNNStateCacheQuantType::NoQuant;

    int kvcacheImplOption = (int)MNNStateCacheType::MNN_STATECACHE_ADVANCED;
};
```

Also, add cases to `Session::ModeGroup` and field to  `Interpreter::HintMode`,
```cpp
void Session::ModeGroup::setHint(Interpreter::HintMode mode, int hint) {
    // ...
    switch (mode) {
        case Interpreter::KVCACHE_QUANT_OPTIONS:
            runtimeHint.kvcacheQuantOption = hint;
            break;
        case Interpreter::KVCACHE_IMPL_OPTIONS:
            runtimeHint.kvcacheImplOption = hint;
            break;
        default:
            break;
    }
}
```

```Cpp
class MNN_PUBLIC StateCacheManager{
    // ...
    void setHint(MNNStateCacheQuantType quantType = MNNStateCacheQuantType::NoQuant, MNNStateCacheType type = MNNStateCacheType::MNN_STATECACHE_ADVANCED);
    // ...
};
```

### Creation of StateCacheManager



### Sampler

#### adding new sampler
1. add hyperparameters to `llm_config.hpp`.
2. add parsing selections to function `void Llm::initSampler()` in `llm.cpp`.
3. add implementation in `sampler.hpp` and `sampler.cpp`.

### Operator

#### Matmul Layout v1.0

Layerwise concatenation of all input blocks. Assume enough memory for activation buffer and enough memory for one layer's calculation.

1. Prepare inputs: concatenate them together.
2. Calculation: perform tiled CPU Matmul.
3. Prepare outputs: disperse the outputs to separate outputs Tensors.