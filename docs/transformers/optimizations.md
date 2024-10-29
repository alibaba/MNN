# Version 1
## v1.0.1

### 1. Compilation
#### 1.1 compile for phone

```bash
# windows
set ANDROID_NDK=D:\NDK\android-ndk
cmake ../.. -DCMAKE_TOOLCHAIN_FILE=%ANDROID_NDK%/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="arm64-v8a" -DANDROID_STL=c++_static -DMNN_USE_LOGCAT=false -DCMAKE_CXX_STANDARD=17 -DMNN_USE_SYSTEM_LIB=OFF -DMNN_BUILD_BENCHMARK=ON -DMNN_USE_SSE=OFF -DMNN_SUPPORT_BF16=OFF -DMNN_BUILD_TEST=ON -DANDROID_NATIVE_API_LEVEL=android-21  -DMNN_BUILD_FOR_ANDROID_COMMAND=true -DNATIVE_LIBRARY_OUTPUT=. -DMNN_LOW_MEMORY=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_SHARED_LIBS=ON
# arm v82 & OPENCL
cmake ../.. -DCMAKE_TOOLCHAIN_FILE=%ANDROID_NDK%/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="arm64-v8a" -DANDROID_STL=c++_static -DMNN_USE_LOGCAT=false -DCMAKE_CXX_STANDARD=17 -DMNN_USE_SYSTEM_LIB=OFF -DMNN_BUILD_BENCHMARK=ON -DMNN_USE_SSE=OFF -DMNN_SUPPORT_BF16=OFF -DMNN_BUILD_TEST=ON -DANDROID_NATIVE_API_LEVEL=android-21  -DMNN_BUILD_FOR_ANDROID_COMMAND=true -DNATIVE_LIBRARY_OUTPUT=. -DMNN_LOW_MEMORY=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_SHARED_LIBS=ON -DMNN_ARM82=ON -DMNN_OPENCL=ON

# linux
export ANDROID_NDK=~/NDK/android-ndk
cmake ../.. -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="arm64-v8a" -DANDROID_STL=c++_static -DMNN_USE_LOGCAT=false -DCMAKE_CXX_STANDARD=17 -DMNN_USE_SYSTEM_LIB=OFF -DMNN_BUILD_BENCHMARK=ON -DMNN_USE_SSE=OFF -DMNN_SUPPORT_BF16=OFF -DMNN_BUILD_TEST=ON -DANDROID_NATIVE_API_LEVEL=android-21  -DMNN_BUILD_FOR_ANDROID_COMMAND=true -DNATIVE_LIBRARY_OUTPUT=. -DMNN_LOW_MEMORY=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_SHARED_LIBS=ON
# arm v82 & OPENCL
cmake ../.. -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="arm64-v8a" -DANDROID_STL=c++_static -DMNN_USE_LOGCAT=false -DCMAKE_CXX_STANDARD=17 -DMNN_USE_SYSTEM_LIB=OFF -DMNN_BUILD_BENCHMARK=ON -DMNN_USE_SSE=OFF -DMNN_SUPPORT_BF16=OFF -DMNN_BUILD_TEST=ON -DANDROID_NATIVE_API_LEVEL=android-21  -DMNN_BUILD_FOR_ANDROID_COMMAND=true -DNATIVE_LIBRARY_OUTPUT=. -DMNN_LOW_MEMORY=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_SHARED_LIBS=ON -DMNN_ARM82=ON -DMNN_OPENCL=ON
```

#### 1.2 compile for pc
LLM and TRANSFORMER_FUSE shall be set ON: `-DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON`

```bash
mkdir build && mkdir build/pc && cd build/pc
# linux
cmake ../.. -DCMAKE_CXX_STANDARD=17 -DMNN_USE_SYSTEM_LIB=OFF -DMNN_BUILD_SHARED_LIBS=ON -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON -DMNN_BUILD_CONVERTER=ON -DMNN_PORTABLE_BUILD=ON -DTFMODEL_OPTIMIZE=ON -DMNN_LOW_MEMORY=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_TEST=ON -DMNN_BUILD_OPENCV=ON 
# windows
cmake ../.. -DCMAKE_CXX_STANDARD=17 -DMNN_USE_SYSTEM_LIB=OFF -DMNN_BUILD_SHARED_LIBS=ON -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON -DMNN_BUILD_CONVERTER=ON -DMNN_PORTABLE_BUILD=ON -DTFMODEL_OPTIMIZE=ON -DMNN_LOW_MEMORY=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_TEST=ON -DMNN_SEP_BUILD=ON -DMNN_BUILD_OPENCV=ON
make -j20

./llm_demo ../../model/qwen1_5-4b-chat-mnn-f/llm_config.json
```

### 2. Model Convertion

#### 2.1 Direct Model Download
Transformer path: 
Download from https://github.com/wangzhaode/mnn-llm/releases

#### 2.2 Model Export
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
        --onnx_path ../../../model/qwen1_5-4b-chat-onnx \
        --mnn_path  ../../../model/qwen1_5-4b-chat-mnn

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
    ./build/MNNConvert -f ONNX --modelFile ./model/qwen1_5-4b-chat-onnx/block_${i}.onnx --MNNModel ./model/qwen1_5-4b-chat-mnn/block_${i}.mnn --weightQuantBits 4 --weightQuantAsymmetric --transformerFuse
done

for i in $(seq 0 24)
do
    ./build/pc/MNNConvert -f ONNX --modelFile ./model/qwen1_5-1_8b-chat-onnx/block_${i}.onnx --MNNModel ./model/qwen1_5-1_8b-chat-mnn/block_${i}.mnn --weightQuantBits 4 --weightQuantAsymmetric --transformerFuse
done
```


### 3. StateCacheManager

Implemented in `core/StateCacheManager.hpp` and `core/StateCacheManager.cpp`.

#### 3.1 Config

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

#### 3.2 Creation and Clone



### 4. Sampler

#### 4.1 sampler specification
llama.cpp sampler support.
sampling:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order:
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature

#### 4.2 sampler references

#### 4.3 process of adding new sampler
1. add hyperparameters to `llm_config.hpp`.
2. add parsing selections to function `void Llm::initSampler()` in `llm.cpp`.
3. add implementation in `sampler.hpp` and `sampler.cpp`.

### 5. CPUAttention

#### 5.1 Matmul Layout v1.0

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

#### 5.2 Speed Evaluation of CPUAttention

1. test case 1
Processor: HUAWEI Kirin 990 5G
Mem: 8GB + 2GB Swap
Model: Qwen1.5-4B-Chat
Precision: attn-fp32, mlp-W4A8
"type_kv": 1 (8-8)

| Phase | prev_seq_len | seq_len | Layer time (ms) | Attention time (ms) | proportion|
| :---: | :---: | :---: | :-----: | :----: | :---: |
| prefill | 0 | 9 | 24.987 | 0.195 | 0.78% |
| prefill | 0 | 18 | 33.978 | 0.333 | 0.98% |
| decode | 10 | 1 | 4.261 | 0.058 | 1.4% |
| decode | 20 | 1 | 4.187 | 0.086 | 2% |
| decode | 60 | 1 | 4.019 | 0.147 | 3.6% |
| decode | 64(malloc) | 1 | 4.005 | 0.222 | 5.5% |
| decode | 80 | 1 | 4.062 | 0.207 | 5.1% |

"type_kv": 0 (63-8)

| Phase | prev_seq_len | seq_len | Layer time (ms) | Attention time (ms) | proportion|
| :---: | :---: | :---: | :-----: | :----: | :---: |
| prefill | 0 | 9 | 25.141 | 0.262 | 1.0% |
| prefill | 0 | 18 | 33.742 | 0.394 | 1.2% |
| decode | 10 | 1 | 4.093 | 0.064 | 1.6% |
| decode | 20 | 1 | 4.163 | 0.073 | 1.75% |
| decode | 60 | 1 | 4.141 | 0.101 | 2.4% |
| decode | 80 | 1 | 4.131 | 0.11 | 2.7% |
| decode | 89(malloc) | 1 | 6.382 | 1.42 | 22.2% |
| decode | 110 | 1 | 4.101 | 0.167 | 4.0% |

2. test case 2

Processor: intel i9-13900
Mem: 32GB + 12GB Swap
Model: Qwen1.5-4B-Chat
Precision: attn-fp32, mlp-W4A8
"type_kv": 1 (24-3)

Q: what's gpu?
A: GPU, or Graphics Processing Unit, is a type of processor that is specifically designed for handling graphics and mathematical calculations. It is commonly used in gaming consoles, high-end graphics workstations, and other applications that require high levels of performance in these areas. The GPU is able to perform these tasks much faster than a traditional CPU, making it an essential component in modern computing systems.

Q: what's cpu?
A: CPU, or Central Processing Unit, is a type of processor that is responsible for executing instructions and performing calculations. It is the primary component of a computer system, and it is responsible for managing all of the system's tasks, including running applications, managing data, and performing calculations. The CPU is able to perform these tasks much faster than a traditional GPU, making it an essential component in modern computing systems.

Q: how do they differ from each other?
A: GPU and CPU are two different types of processors that are used in different types of computing systems. The main difference between them is that GPU is specifically designed for handling graphics and mathematical calculations, while CPU is responsible for executing instructions and performing calculations. The GPU is able to perform these tasks much faster than a traditional CPU, making it an essential component in modern computing systems. The CPU, on the other hand, is able to perform these tasks much faster than a traditional GPU, making it an essential component in modern computing systems.

| Phase | prev_seq_len | seq_len | Layer time (ms) | Attention time (ms) | proportion (%) |
| :---: | :---: | :---: | :-----: | :----: | :---: |
| prefill | 0 | 18 | 18.5903 | 0.1532 | 0.8238 |
| decode | 30 | 1 | 2.9778 | 0.0569 | 1.9116 |
| decode | 60 | 1 | 2.2643 | 0.0702 | 3.0980 |
| decode | 64 | 1 | 2.8403 | 0.0731 | 2.5745 |
| decode | 72(malloc) | 1 | 2.7708 | 0.1302 | 4.6972 |
| decode | 78 | 1 | 2.4339 | 0.0853 | 3.5026 |
| decode | 80 | 1 | 2.6721 | 0.0871 | 3.2606 |
| decode | 82 | 1 | 2.3748 | 0.0824 | 3.4698 |
| decode | 88 | 1 | 2.5643 | 0.0895 | 3.4882 |
| decode | 108 | 1 | 1.9229 | 0.1038 | 5.3993 |
| decode | 119 | 1 | 2.2829 | 0.1093 | 4.7878 |
| decode | 120(new) | 1 | 2.3890 | 0.1534 | 6.4201 |
| decode | 140 | 1 | 2.8935 | 0.1347 | 4.6561 |
| decode | 144(malloc) | 1 | 2.9495 | 0.2027 | 6.8714 |
| decode | 160 | 1 | 2.5395 | 0.1353 | 5.3288 |
| decode | 180(new) | 1 | 2.7440 | 0.1890 | 6.8895 |
| decode | 210 | 1 | 2.5207 | 0.1857 | 7.3670 |
| decode | 216(malloc) | 1 | 2.4529 | 0.2630 | 10.7211 |
| decode | 220 | 1 | 2.2560 | 0.1847 | 8.1872 |
| decode | 240(new) | 1 | 2.6265 | 0.2428 | 9.2423 |
| decode | 280 | 1 | 2.0357 | 0.2051 | 10.0775 |
| decode | 288(malloc) | 1 | 2.1922 | 0.2640 | 12.0404 |
| decode | 300 | 1 | 2.4628 | 0.2314 | 9.3957 |

"type_kv": 0 (63-24)

Q: what's gpu?
A: GPU, or Graphics Processing Unit, is a type of hardware that is specifically designed to perform mathematical calculations and graphical tasks. It is commonly used in applications such as gaming, video editing, and scientific simulations. The GPU is able to perform these tasks much faster than a traditional CPU, which is responsible for handling general-purpose tasks such as file I/O and web browsing. The GPU is typically connected to the CPU via a graphics card, which is a specialized device that is designed to handle graphics tasks.

Q: what's cpu?
A: CPU, or Central Processing Unit, is a type of hardware that is specifically designed to perform general-purpose tasks such as file I/O, web browsing, and scientific simulations. It is commonly used in applications such as gaming, video editing, and scientific simulations. The CPU is able to perform these tasks much faster than a traditional GPU, which is responsible for handling graphics tasks. The CPU is typically connected to the GPU via a graphics card, which is a specialized device that is designed to handle graphics tasks.

Q: how do they differ from each other?
A: GPUs and CPUs differ from each other in several ways. The most significant difference is that GPUs are specifically designed to handle graphics tasks, while CPUs are specifically designed to handle general-purpose tasks. This means that GPUs are much faster than CPUs when it comes to performing graphics tasks, such as rendering images and videos. CPUs, on the other hand, are much faster than GPUs when it comes to performing general-purpose tasks, such as file I/O and web browsing.
GPUs are also able to perform certain tasks much faster than CPUs, such as matrix multiplication and vector addition. This is because GPUs are able to perform these tasks using a technique called "parallel processing," which involves dividing a task into smaller pieces and performing each piece in parallel. CPUs, on the other hand, are able to perform these tasks using a technique called "serial processing," which involves performing a task one piece at a time.
GPUs are also able to perform certain tasks much faster than CPUs, such as

| Phase | prev_seq_len | seq_len | Layer time (ms) | Attention time (ms) | proportion (%) |
| :---: | :---: | :---: | :-----: | :----: | :---: |
| prefill | 0 | 18 | 18.5213 | 0.1994 | 1.0767 |
| decode | 30 | 1 | 3.0043 | 0.0592 | 1.9697 |
| decode | 60 | 1 | 2.4122 | 0.0641 | 2.6552 |
| decode | 64 | 1 | 2.2182 | 0.0640 | 2.8830 |
| decode | 72 | 1 | 2.5079 | 0.0739 | 2.9467 |
| decode | 78 | 1 | 2.5453 | 0.0689 | 2.7070 |
| decode | 80 | 1 | 2.1048 | 0.0694 | 3.2948 |
| decode | 82 | 1 | 2.2666 | 0.0668 | 2.9483 |
| decode | 88 | 1 | 2.5103 | 0.0788 | 3.1380 |
| decode | 96(re-alloc) | 1 | 3.8289 | 1.2392 | 32.3657 |
| decode | 100 | 1 | 2.7265 | 0.0884 | 3.2404 |
| decode | 140 | 1 | 2.5288 | 0.1024 | 4.0483 |
| decode | 144 | 1 | 2.9095 | 0.1088 | 3.7377 |
| decode | 160 | 1 | 2.6942 | 0.1125 | 4.1737 |
| decode | 168(re-alloc) | 1 | 3.7834 | 1.1970 | 31.6375 |
| decode | 180 | 1 | 2.7585 | 0.1271 | 4.6076 |
| decode | 200 | 1 | 2.4005 | 0.1234 | 5.1385 |
| decode | 220 | 1 | 2.4310 | 0.1324 | 5.4473 |
| decode | 280 | 1 | 2.4475 | 0.1602 | 6.5476 |
| decode | 288 | 1 | 2.5880 | 0.1681 | 6.4964 |
| decode | 300 | 1 | 2.7049 | 0.1711 | 6.3264 |
| decode | 312(re-alloc) | 1 | 5.7744 | 2.1637 | 37.4697 |
| decode | 320 | 1 | 2.4353 | 0.1894 | 7.7772 |
| decode | 360 | 1 | 3.6315 | 0.2687 | 7.3985 |
| decode | 384(re-alloc) | 1 | 6.2943 | 2.5750 | 40.9104 |
| decode | 400 | 1 | 3.4962 | 0.2542 | 7.2700 |



100 6.20021
200 6.13895
300 6.05897
400 5.99104
500 5.91477
600 5.83586

100 6.25798
200 6.16745
300 6.09691
400 6.03272
500 5.9634
600 5.90768

block_size = 64
assume one 220 us 
addition: ~0.7 us/block, normal allocation: 3~4 us
q @ k: 5 blocks, 16 us vs 13 us
qk @ v: 5 blocks, 
transform+mask+softmax, 300 tokens: 7us