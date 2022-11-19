#include "LayerNormExecution.hpp"
namespace MNN {
namespace CUDA {

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)



#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}

template <typename T>
__global__ 
void input_layernorm(T* out, const T* input, const float* gamma, const float* beta, int m, int n, const float epsilon, int sumPerKnl)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;

  for(int idx=0; idx<sumPerKnl && idx*256 + tid < n; idx++) {
    local_out += (float)(input[blockIdx.x * n + idx*256 + tid]);
  }

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float var_tmp = 0.0f;
  for(int idx=0; idx<sumPerKnl && idx*256 + tid < n; idx++) {
    var_tmp += (((float)input[blockIdx.x * n + idx*256 + tid] - s_mean) * ((float)input[blockIdx.x * n + idx*256 + tid] - s_mean));
  }
  variance += blockReduceSum<float>(var_tmp);
  if(threadIdx.x == 0)
    s_variance = variance / n + epsilon;
  __syncthreads();

  for(int idx=0; idx<sumPerKnl && idx*256 + tid < n; idx++) {
    float res = (((float)input[blockIdx.x * n + idx*256 + tid] - s_mean) * rsqrtf(s_variance));
    if(gamma != nullptr && beta != nullptr) {
        res = res * (float)(__ldg(&gamma[idx*256 + tid])) + (float)(__ldg(&beta[idx*256 + tid]));
    }
    out[blockIdx.x * n + idx*256+tid] = (T)res;
  }
}


template <typename T>
__global__ 
void input_layernorm_2048(T* out, const T* input, const float* gamma, const float* beta, int m, int n, const float epsilon)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;

  float value_tmp[8];
  value_tmp[0] = input[blockIdx.x * 2048 + 0*256 + tid];
  value_tmp[1] = input[blockIdx.x * 2048 + 1*256 + tid];
  value_tmp[2] = input[blockIdx.x * 2048 + 2*256 + tid];
  value_tmp[3] = input[blockIdx.x * 2048 + 3*256 + tid];
  value_tmp[4] = input[blockIdx.x * 2048 + 4*256 + tid];
  value_tmp[5] = input[blockIdx.x * 2048 + 5*256 + tid];
  value_tmp[6] = input[blockIdx.x * 2048 + 6*256 + tid];
  value_tmp[7] = input[blockIdx.x * 2048 + 7*256 + tid];

  #pragma unroll(8)
  for(int idx=0; idx<8; idx++) {
    local_out += (float)value_tmp[idx];
  }

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float var_tmp = 0.0f;

  #pragma unroll(8)
  for(int idx=0; idx<8; idx++) {
    var_tmp += ((value_tmp[idx] - s_mean) * (value_tmp[idx] - s_mean));
  }
  variance += blockReduceSum<float>(var_tmp);
  if(threadIdx.x == 0)
    s_variance = variance / n + epsilon;
  __syncthreads();

  #pragma unroll(8)
  for(int idx=0; idx<8; idx++) {
    float res = ((value_tmp[idx] - s_mean) * rsqrtf(s_variance));
    if(gamma != nullptr && beta != nullptr) {
        res = res * (float)(__ldg(&gamma[idx*256 + tid])) + (float)(__ldg(&beta[idx*256 + tid]));
    }
    out[blockIdx.x * 2048 + idx*256+tid] = (T)res;
  }
}


template <typename T>
__global__ 
void input_layernorm_1024(T* out, const T* input, const float* gamma, const float* beta, int m, int n, const float epsilon)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;

  float value_tmp[4];
  value_tmp[0] = input[blockIdx.x * 1024 + 0*256 + tid];
  value_tmp[1] = input[blockIdx.x * 1024 + 1*256 + tid];
  value_tmp[2] = input[blockIdx.x * 1024 + 2*256 + tid];
  value_tmp[3] = input[blockIdx.x * 1024 + 3*256 + tid];

  #pragma unroll(4)
  for(int idx=0; idx<4; idx++) {
    local_out += (float)value_tmp[idx];
  }

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float var_tmp = 0.0f;

  #pragma unroll(4)
  for(int idx=0; idx<4; idx++) {
    var_tmp += ((value_tmp[idx] - s_mean) * (value_tmp[idx] - s_mean));
  }
  variance += blockReduceSum<float>(var_tmp);
  if(threadIdx.x == 0)
    s_variance = variance / n + epsilon;
  __syncthreads();

  #pragma unroll(4)
  for(int idx=0; idx<4; idx++) {
    float res = ((value_tmp[idx] - s_mean) * rsqrtf(s_variance));
    if(gamma != nullptr && beta != nullptr) {
        res = res * (float)(__ldg(&gamma[idx*256 + tid])) + (float)(__ldg(&beta[idx*256 + tid]));
    }
    out[blockIdx.x * 1024 + idx*256+tid] = (T)res;
  }
}


template <typename T>
__global__ 
void input_layernorm_512(T* out, const T* input, const float* gamma, const float* beta, int m, int n, const float epsilon)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;

  float value_tmp[2];
  value_tmp[0] = input[blockIdx.x * 512 + 0*256 + tid];
  value_tmp[1] = input[blockIdx.x * 512 + 1*256 + tid];

  local_out += (float)value_tmp[0];
  local_out += (float)value_tmp[1];

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float var_tmp = 0.0f;
  var_tmp += ((value_tmp[0] - s_mean) * (value_tmp[0] - s_mean));
  var_tmp += ((value_tmp[1] - s_mean) * (value_tmp[1] - s_mean));

  variance += blockReduceSum<float>(var_tmp);
  if(threadIdx.x == 0)
    s_variance = variance / n + epsilon;
  __syncthreads();

  float res0 = ((value_tmp[0] - s_mean) * rsqrtf(s_variance));
  float res1 = ((value_tmp[1] - s_mean) * rsqrtf(s_variance));

  if(gamma != nullptr && beta != nullptr) {
      res0 = res0 * (float)(__ldg(&gamma[0*256 + tid])) + (float)(__ldg(&beta[0*256 + tid]));
      res1 = res1 * (float)(__ldg(&gamma[1*256 + tid])) + (float)(__ldg(&beta[1*256 + tid]));
  }

  out[blockIdx.x * 512 + 0*256+tid] = (T)res0;
  out[blockIdx.x * 512 + 1*256+tid] = (T)res1;
}


template<typename T>
__global__ void LAYERNORM(const int count, const int outside, const int inside, const float epsilon, 
                          const T* in, T* out, const float* gamma_data, const float* beta_data) {
    CUDA_KERNEL_LOOP(i, count) {
        const int o = i / inside;
        const int index = i % inside;
        const T* inner_input = in + o * inside;
        T* inner_output = out + o * inside;
        float sum = 0.f;
        for (int j = 0; j < inside; ++j) {
            sum += (float)inner_input[j];
        }
        float mean = sum / inside;
        float square_sum = 0.f;
        for (int j = 0; j < inside; ++j) {
            square_sum += ((float)inner_input[j] - mean) * ((float)inner_input[j] - mean);
        }
        float variable = square_sum / inside;
        variable = 1.f / sqrt(variable + epsilon);

        float res = ((float)inner_input[index] - mean) * variable;
        if(gamma_data != nullptr && beta_data != nullptr) {
            res = res * gamma_data[index] + beta_data[index];
        }
        inner_output[index] = (T)res;
    }
}

LayerNormExecution::LayerNormExecution(const LayerNorm* layer_norm_param, Backend *backend) : Execution(backend) {
    int axis_size = layer_norm_param->axis()->size();
    mAxises.resize(axis_size);
    for (int i = 0; i < axis_size; ++i) {
        mAxises[i] = layer_norm_param->axis()->Get(i);
    }

    mEps = layer_norm_param->epsilon();
    mGroup = layer_norm_param->group();

    if (layer_norm_param->gamma() && layer_norm_param->beta()) {
        int size = layer_norm_param->gamma()->size();
        mGammaTensor.reset(Tensor::createDevice<int32_t>({size}));
        auto status = backend->onAcquireBuffer(mGammaTensor.get(), Backend::STATIC);
        if (!status) {
            MNN_ERROR("Out of memory when gamma is acquired in CudaLayerNorm.\n");
        }

        mDeviceGamma = (void *)mGammaTensor.get()->buffer().device;
        const float* gamma_data = layer_norm_param->gamma()->data();
        cudaMemcpy(mDeviceGamma, gamma_data, size * sizeof(float), cudaMemcpyHostToDevice);

        if (layer_norm_param->beta()->size() != size) {
            MNN_ERROR("Size of gamma and beta are not match in CudaLayerNorm.\n");
        }
        mBetaTensor.reset(Tensor::createDevice<int32_t>({size}));
        status = backend->onAcquireBuffer(mBetaTensor.get(), Backend::STATIC);
        if (!status) {
            MNN_ERROR("Out of memory when beta is acquired in CudaLayerNorm.\n");
        }

        mDeviceBeta = (void *)mBetaTensor.get()->buffer().device;
        const float* beta_data = layer_norm_param->beta()->data();
        cudaMemcpy(mDeviceBeta, beta_data, size * sizeof(float), cudaMemcpyHostToDevice);
    }
}
LayerNormExecution::~LayerNormExecution() {
    // Do nothing
}

ErrorCode LayerNormExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];

    mOutside = 1;
    mInside = 1;
    int rank = input->dimensions();
    if (mGroup > 1) {
        mOutside = input->length(0) * mGroup;
        for (int i = 1; i < rank; i++) {
            mInside *= input->length(i);
        }
        mInside /= mGroup;
        return NO_ERROR;
    }
    std::vector<int> axis(mAxises.size());
    for (int i = 0; i < mAxises.size(); ++i) {
        if (mAxises[i] < 0) {
            mAxises[i] += rank;
        }
    }
    std::sort(axis.begin(), axis.end());
    for (int i = 0; i < rank - axis.size(); ++i) {
        mOutside *= input->length(i);
    }
    for (int i = rank - axis.size(); i < rank; ++i) {
        mInside *= input->length(i);
    }

    return NO_ERROR;
}

ErrorCode LayerNormExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
 
    int block_num = runtime->blocks_num(mOutside*mInside);
    int threads_num = runtime->threads_num();
    auto input_addr = (void*)inputs[0]->deviceId();
    auto output_addr = (void*)outputs[0]->deviceId();
    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        if(mInside < 128) {
            LAYERNORM<<<block_num, threads_num>>>(mOutside*mInside, mOutside, mInside, mEps, (const half *)input_addr, (half *)output_addr,
                    (const float *)mDeviceGamma, (const float *)mDeviceBeta);
        } else {
            if(mInside == 2048) {
                input_layernorm_2048<<<mOutside, 256>>>((half *)output_addr, (const half *)input_addr, (const float *)mDeviceGamma, 
                    (const float *)mDeviceBeta, mOutside, mInside, mEps);
            } else if(mInside == 1024) {
                input_layernorm_1024<<<mOutside, 256>>>((half *)output_addr, (const half *)input_addr, (const float *)mDeviceGamma, 
                    (const float *)mDeviceBeta, mOutside, mInside, mEps);
            } else if(mInside == 512) {
                input_layernorm_512<<<mOutside, 256>>>((half *)output_addr, (const half *)input_addr, (const float *)mDeviceGamma, 
                    (const float *)mDeviceBeta, mOutside, mInside, mEps);
            } else {
                int sumPerKnl = (mInside+255) / 256;
                input_layernorm<<<mOutside, 256>>>((half *)output_addr, (const half *)input_addr, (const float *)mDeviceGamma, 
                    (const float *)mDeviceBeta, mOutside, mInside, mEps, sumPerKnl);
            }
        }
        return NO_ERROR;
    }

    if(mInside < 128) {
        LAYERNORM<<<block_num, threads_num>>>(mOutside*mInside, mOutside, mInside, mEps, (const float *)input_addr, (float *)output_addr,
                (const float *)mDeviceGamma, (const float *)mDeviceBeta);
    } else {
        if(mInside == 2048) {
            input_layernorm_2048<<<mOutside, 256>>>((float *)output_addr, (const float *)input_addr, (const float *)mDeviceGamma, 
                (const float *)mDeviceBeta, mOutside, mInside, mEps);
        } else if(mInside == 1024) {
            input_layernorm_1024<<<mOutside, 256>>>((float *)output_addr, (const float *)input_addr, (const float *)mDeviceGamma, 
                (const float *)mDeviceBeta, mOutside, mInside, mEps);
        } else if(mInside == 512) {
            input_layernorm_512<<<mOutside, 256>>>((float *)output_addr, (const float *)input_addr, (const float *)mDeviceGamma, 
                (const float *)mDeviceBeta, mOutside, mInside, mEps);
        } else {
            int sumPerKnl = (mInside+255) / 256;
            input_layernorm<<<mOutside, 256>>>((float *)output_addr, (const float *)input_addr, (const float *)mDeviceGamma, 
                (const float *)mDeviceBeta, mOutside, mInside, mEps, sumPerKnl);
        }
    }
    return NO_ERROR;
}

class LayerNormCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_LayerNorm();
        return new LayerNormExecution(param, backend);
    }
};

static CUDACreatorRegister<LayerNormCreator> __init(OpType_LayerNorm);

}
}
