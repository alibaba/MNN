## StateCacheManager

### Compilation
#### compile for phone

```bash
set ANDROID_NDK=D:\NDK\android-ndk-r26d
cmake ../.. -DCMAKE_TOOLCHAIN_FILE=%ANDROID_NDK%/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="arm64-v8a" -DANDROID_STL=c++_static -DMNN_USE_LOGCAT=false -DCMAKE_CXX_STANDARD=17 -DMNN_USE_SYSTEM_LIB=OFF -DMNN_BUILD_BENCHMARK=ON -DMNN_USE_SSE=OFF -DMNN_SUPPORT_BF16=OFF -DMNN_BUILD_TEST=ON -DANDROID_NATIVE_API_LEVEL=android-21  -DMNN_BUILD_FOR_ANDROID_COMMAND=true -DNATIVE_LIBRARY_OUTPUT=. -DMNN_LOW_MEMORY=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_SHARED_LIBS=ON

export ANDROID_NDK=~/NDK/android-ndk-r27
cmake ../.. -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="arm64-v8a" -DANDROID_STL=c++_static -DMNN_USE_LOGCAT=false -DCMAKE_CXX_STANDARD=17 -DMNN_USE_SYSTEM_LIB=OFF -DMNN_BUILD_BENCHMARK=ON -DMNN_USE_SSE=OFF -DMNN_SUPPORT_BF16=OFF -DMNN_BUILD_TEST=ON -DANDROID_NATIVE_API_LEVEL=android-21  -DMNN_BUILD_FOR_ANDROID_COMMAND=true -DNATIVE_LIBRARY_OUTPUT=. -DMNN_LOW_MEMORY=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_SHARED_LIBS=ON
```

#### compile for pc
LLM and TRANSFORMER_FUSE shall be set ON: `-DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON`

```bash
cmake ../.. -DCMAKE_CXX_STANDARD=17 -DMNN_USE_SYSTEM_LIB=OFF -DMNN_BUILD_SHARED_LIBS=ON -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON -DMNN_BUILD_CONVERTER=ON -DMNN_PORTABLE_BUILD=ON -DTFMODEL_OPTIMIZE=ON -DMNN_LOW_MEMORY=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_TEST=ON
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

python llm_export.py \
        --path ../../../model/Qwen1_5-1_8B-Chat \
        --type Qwen1_5-1_8B-Chat \
        --export_split \
        --export_token \
        --export_mnn \
        --export_embed --embed_bin --embed_bf16 \
        --onnx_path ../../../model/qwen1_5-1_8b-chat-onnx \
        --mnn_path  ../../../model/qwen1_5-1_8b-chat-mnn
```

Currently fuse attention haven't added to python MNN package converter. To use it, you need directly use MNNConvert.
```bash
for i in $(seq 0 39)
do
    ./build/MNNConvert -f ONNX --modelFile ./model/qwen1_5-4b-chat-onnx/block_${i}.onnx --MNNModel ./model/qwen1_5-4b-chat-mnn-f/block_${i}.mnn --weightQuantBits 4 --weightQuantAsymmetric --transformerFuse
done

for i in $(seq 0 24)
do
    ./build/pc/MNNConvert -f ONNX --modelFile ./model/qwen1_5-1_8b-chat-onnx/block_${i}.onnx --MNNModel ./model/qwen1_5-1_8b-chat-mnn/block_${i}.mnn --weightQuantBits 4 --weightQuantAsymmetric --transformerFuse
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

For StateCacheBlock:
- Tensor shape: 
  - K: [kvnum_heads, block_size / hp, head_dim, hp], 
  - V: [kvnum_heads, head_dim / hp, block_size, hp]
-  block_size % hp == 0, block_size % core->pack == 0 
- mSlotNum: mSlotNum shall not be modified in thread to ensure thread-safe. Besides, no lock is ever implemented.


1. Prepare inputs: pack new k and v into pastKV (StateCacheBlock).
2. Calculation: perform multi-threaded tiled CPU Matmul.
3. Prepare outputs: pack the dispersed outputs to one outputs Tensors.
4. Update pastKV mSlotNum after all calculation threads are finished.


In the future, modify the operators to enable fused flash attention, further curbing memory buffer size and computation time.

q @ k: q @ [k1, k2, k3, k4] = [q @ k1, q @ k2 + q @ k3 + q @ k4]

qk @ v: [a1, a2, a3, a4] @ [b1, b2, b3, b4]^T = a1 @ b1 + a2 @ b2 + a3 @ b3 + a4 @ b4
(ai @ bi are all of the same shape of qk @ v)
(qk shall be dispersed and v shall be dispersed)


To Do List:
- [x] dispersed pastKV
- [x] dispersed packQK
- [x] dispersed packQKV
- [x] pack_key
- [x] pack_value
- [x] query @ key
- [x] unpack_QK (for softmax)
- [x] pack_QK (for qk @ v)
- [x] dequant_value_fp16
- [x] dequant_value_float
- [x] qk @ v
- [x] update mSlotNum for all blocks


Coding Cautions:

- When creating a new backend, make sure that they share the very same StateCacheManager. This happens during load() and clone() of a Module.

- std::shared_ptr<StateCacheManager> is only set in global Executor, while all the other pointers are simple naive pointer.

- We keep lots of pointers, but only the malloc one can be freed, because of an a block area indicating the block size for following free().

- Remember to flush when debugging with print.

- finish single-threaded debugging first, and then multi-threading.

- take good care of memory layout please.

- look at the code! Not the doc nor function names!

- realize the functionality of std::move



```cpp
// mPackQ
mPackQ.reset(Tensor::createDevice<float>({mThreadNum, UP_DIV(seq_len, eP), mResource->mHeadDim, eP}));
// mPackQKV
mPackQKV.reset(Tensor::createDevice<float>({mThreadNum, UP_DIV(mResource->mHeadDim, unit), seq_len, unit}));
// packQK
// length: pastKV.size()
packQK.emplace_back(Tensor::createDevice<float>({mThreadNum, UP_DIV(block_size, unit), seq_len, unit}));
// pack_q
auto pack_q      = mPackQ->host<char>() + tId * UP_DIV(seq_len, eP) * mResource->mHeadDim * eP * bytes;
auto pack_qk = packQK[j]->host<char>() + tId * UP_DIV(block_size, unit) * seq_len * unit * bytes;

// Tensor shape: 
//     K: [num_heads, block_size / hP, head_dim, hP]
//     V: [num_heads, head_dim / hP, block_size, hP]
std::shared_ptr<Tensor> unpackQK(Tensor::createDevice<int32_t>({mThreadNum, seq_len, kv_seq_len}));
auto unpack_qk   = unpackQK->host<float>() + tId * seq_len * kv_seq_len;


// newPackQK.emplace_back(Tensor::createDevice<float>({mThreadNum, UP_DIV(seq_len, eP), block_size, eP}));
// newPackQK.emplace_back(Tensor::createDevice<float>({mThreadNum, UP_DIV(seq_len, eP), last_block_slot_num, eP}));

std::shared_ptr<Tensor> newPackQK(Tensor::createDevice<float>({mThreadNum, UP_DIV(seq_len, eP), kv_seq_len, eP}));
std::shared_ptr<Tensor> VBuffer(Tensor::createDevice<float>({mThreadNum, UP_DIV(mResource->mHeadDim, hP), kv_seq_len, hP}));


std::shared_ptr<Tensor> QKVBuffer(Tensor::createDevice<float>({mThreadNum, UP_DIV(mResource->mHeadDim, unit), seq_len, unit}));
char* qkv_buffer = QKVBuffer->host<char>() + tId * UP_DIV(mResource->mHeadDim, unit) * seq_len * unit * bytes;


mPackQKV.reset(Tensor::createDevice<float>({mThreadNum, UP_DIV(mResource->mHeadDim, unit), seq_len, unit}));
auto pack_qkv    = mPackQKV->host<char>() + tId * UP_DIV(mResource->mHeadDim, unit) * seq_len * unit * bytes;
```
