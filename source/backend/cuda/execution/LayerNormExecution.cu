#include "LayerNormExecution.hpp"
namespace MNN {
namespace CUDA {

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename T>
__global__ 
void input_layernorm(T* out, const T* input, const float* gamma, const float* beta, int m, int n, const float epsilon, int sumPerKnl, bool RMSNorm)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;

  if(!RMSNorm){
    for(int idx=0; idx<sumPerKnl && idx*256 + tid < n; idx++) {
     local_out += (float)(input[blockIdx.x * n + idx*256 + tid]);
    }

    mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
      s_mean = mean / n;
    __syncthreads();
  }
  mean = s_mean;

  float var_tmp = 0.0f;
  for(int idx=0; idx<sumPerKnl && idx*256 + tid < n; idx++) {
    var_tmp += (((float)input[blockIdx.x * n + idx*256 + tid] - mean) * ((float)input[blockIdx.x * n + idx*256 + tid] - mean));
  }
  variance += blockReduceSum<float>(var_tmp);
  if(threadIdx.x == 0)
    s_variance = variance / n + epsilon;
  __syncthreads();

  for(int idx=0; idx<sumPerKnl && idx*256 + tid < n; idx++) {
    float res = (((float)input[blockIdx.x * n + idx*256 + tid] - mean) * rsqrtf(s_variance));
    if(gamma != nullptr && beta != nullptr) {
        res = res * (float)(__ldg(&gamma[idx*256 + tid])) + (float)(__ldg(&beta[idx*256 + tid]));
    }
    out[blockIdx.x * n + idx*256+tid] = (T)res;
  }
}

template <typename T>
__global__ 
void input_layernorm_320(T* out, const T* input, const float* gamma, const float* beta, int m, int n, const float epsilon, bool RMSNorm)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;

  float value_tmp[5];
  value_tmp[0] = input[blockIdx.x * n + 0*64 + tid];
  value_tmp[1] = input[blockIdx.x * n + 1*64 + tid];
  value_tmp[2] = input[blockIdx.x * n + 2*64 + tid];
  value_tmp[3] = input[blockIdx.x * n + 3*64 + tid];
  value_tmp[4] = input[blockIdx.x * n + 4*64 + tid];

  if(!RMSNorm){
    for(int idx=0; idx<5; idx++) {
     local_out += value_tmp[idx];
    }

    mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
      s_mean = mean / n;
    __syncthreads();
  }
  mean = s_mean;

  float var_tmp = 0.0f;
  for(int idx=0; idx<5; idx++) {
    var_tmp += ((value_tmp[idx] - mean) * (value_tmp[idx] - mean));
  }
  variance += blockReduceSum<float>(var_tmp);
  if(threadIdx.x == 0)
    s_variance = variance / n + epsilon;
  __syncthreads();

  for(int idx=0; idx<5; idx++) {
    float res = ((value_tmp[idx] - mean) * rsqrtf(s_variance));
    if(gamma != nullptr && beta != nullptr) {
        res = res * (float)(__ldg(&gamma[idx*64 + tid])) + (float)(__ldg(&beta[idx*64 + tid]));
    }
    out[blockIdx.x * n + idx*64+tid] = (T)res;
  }
}


template <typename T>
__global__ 
void input_layernorm_2048(T* out, const T* input, const float* gamma, const float* beta, int m, int n, const float epsilon, bool RMSNorm)
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

  if(!RMSNorm){
    #pragma unroll(8)
    for(int idx=0; idx<8; idx++) {
      local_out += (float)value_tmp[idx];
    }

    mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
     s_mean = mean / n;
    __syncthreads();
  }
  mean = s_mean;

  float var_tmp = 0.0f;

  #pragma unroll(8)
  for(int idx=0; idx<8; idx++) {
    var_tmp += ((value_tmp[idx] - mean) * (value_tmp[idx] - mean));
  }
  variance += blockReduceSum<float>(var_tmp);
  if(threadIdx.x == 0)
    s_variance = variance / n + epsilon;
  __syncthreads();

  #pragma unroll(8)
  for(int idx=0; idx<8; idx++) {
    float res = ((value_tmp[idx] - mean) * rsqrtf(s_variance));
    if(gamma != nullptr && beta != nullptr) {
        res = res * (float)(__ldg(&gamma[idx*256 + tid])) + (float)(__ldg(&beta[idx*256 + tid]));
    }
    out[blockIdx.x * 2048 + idx*256+tid] = (T)res;
  }
}


template <typename T>
__global__ 
void input_layernorm_1024(T* out, const T* input, const float* gamma, const float* beta, int m, int n, const float epsilon, bool RMSNorm)
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

  if(!RMSNorm){
    #pragma unroll(4)
    for(int idx=0; idx<4; idx++) {
      local_out += (float)value_tmp[idx];
   }

   mean = blockReduceSum<float>(local_out);
   if(threadIdx.x == 0)
      s_mean = mean / n;
    __syncthreads();
  }
  mean = s_mean;

  float var_tmp = 0.0f;

  #pragma unroll(4)
  for(int idx=0; idx<4; idx++) {
    var_tmp += ((value_tmp[idx] - mean) * (value_tmp[idx] - mean));
  }
  variance += blockReduceSum<float>(var_tmp);
  if(threadIdx.x == 0)
    s_variance = variance / n + epsilon;
  __syncthreads();

  #pragma unroll(4)
  for(int idx=0; idx<4; idx++) {
    float res = ((value_tmp[idx] - mean) * rsqrtf(s_variance));
    if(gamma != nullptr && beta != nullptr) {
        res = res * (float)(__ldg(&gamma[idx*256 + tid])) + (float)(__ldg(&beta[idx*256 + tid]));
    }
    out[blockIdx.x * 1024 + idx*256+tid] = (T)res;
  }
}


template <typename T>
__global__ 
void input_layernorm_512(T* out, const T* input, const float* gamma, const float* beta, int m, int n, const float epsilon, bool RMSNorm)
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

  if(!RMSNorm){
    local_out += (float)value_tmp[0];
    local_out += (float)value_tmp[1];

    mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
      s_mean = mean / n;
    __syncthreads();
  }
  mean = s_mean;

  float var_tmp = 0.0f;
  var_tmp += ((value_tmp[0] - mean) * (value_tmp[0] - mean));
  var_tmp += ((value_tmp[1] - mean) * (value_tmp[1] - mean));

  variance += blockReduceSum<float>(var_tmp);
  if(threadIdx.x == 0)
    s_variance = variance / n + epsilon;
  __syncthreads();

  float res0 = ((value_tmp[0] - mean) * rsqrtf(s_variance));
  float res1 = ((value_tmp[1] - mean) * rsqrtf(s_variance));

  if(gamma != nullptr && beta != nullptr) {
      res0 = res0 * (float)(__ldg(&gamma[0*256 + tid])) + (float)(__ldg(&beta[0*256 + tid]));
      res1 = res1 * (float)(__ldg(&gamma[1*256 + tid])) + (float)(__ldg(&beta[1*256 + tid]));
  }

  out[blockIdx.x * 512 + 0*256+tid] = (T)res0;
  out[blockIdx.x * 512 + 1*256+tid] = (T)res1;
}


template<typename T>
__global__ void LAYERNORM(const int count, const int outside, const int inside, const float epsilon, 
                          const T* in, T* out, const float* gamma_data, const float* beta_data, bool RMSNorm) {
    CUDA_KERNEL_LOOP(i, count) {
        const int o = i / inside;
        const int index = i % inside;
        const T* inner_input = in + o * inside;
        T* inner_output = out + o * inside;
        float mean = 0.0f;
        if(!RMSNorm){
          float sum = 0.f;
          for (int j = 0; j < inside; ++j) {
            sum += (float)inner_input[j];
          }
          mean = sum / inside;
        }
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
    if (nullptr != layer_norm_param->axis()) {
        mAxises = layer_norm_param->axis()->size();
    }

    mEps = layer_norm_param->epsilon();
    mGroup = layer_norm_param->group();
    RMSNorm = layer_norm_param->useRMSNorm();

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
    for (int i = 0; i < rank - mAxises; ++i) {
        mOutside *= input->length(i);
    }
    for (int i = rank - mAxises; i < rank; ++i) {
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

    //printf("ln:%d-%d\n", mOutside, mInside);
    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        if(mInside < 128) {
            LAYERNORM<<<block_num, threads_num>>>(mOutside*mInside, mOutside, mInside, mEps, (const half *)input_addr, (half *)output_addr,
                    (const float *)mDeviceGamma, (const float *)mDeviceBeta, RMSNorm);
        } else {
            if(mInside == 2048) {
                input_layernorm_2048<<<mOutside, 256>>>((half *)output_addr, (const half *)input_addr, (const float *)mDeviceGamma, 
                    (const float *)mDeviceBeta, mOutside, mInside, mEps, RMSNorm);
            } else if(mInside == 1024) {
                input_layernorm_1024<<<mOutside, 256>>>((half *)output_addr, (const half *)input_addr, (const float *)mDeviceGamma, 
                    (const float *)mDeviceBeta, mOutside, mInside, mEps, RMSNorm);
            } else if(mInside == 512) {
                input_layernorm_512<<<mOutside, 256>>>((half *)output_addr, (const half *)input_addr, (const float *)mDeviceGamma, 
                    (const float *)mDeviceBeta, mOutside, mInside, mEps, RMSNorm);
            } else if(mInside == 320) {
                input_layernorm_320<<<mOutside, 64>>>((half *)output_addr, (const half *)input_addr, (const float *)mDeviceGamma, 
                    (const float *)mDeviceBeta, mOutside, mInside, mEps, RMSNorm);
            } else {
                int sumPerKnl = (mInside+255) / 256;
                input_layernorm<<<mOutside, 256>>>((half *)output_addr, (const half *)input_addr, (const float *)mDeviceGamma, 
                    (const float *)mDeviceBeta, mOutside, mInside, mEps, sumPerKnl, RMSNorm);
            }
        }
        return NO_ERROR;
    }

    if(mInside < 128) {
        LAYERNORM<<<block_num, threads_num>>>(mOutside*mInside, mOutside, mInside, mEps, (const float *)input_addr, (float *)output_addr,
                (const float *)mDeviceGamma, (const float *)mDeviceBeta, RMSNorm);
    } else {
        if(mInside == 2048) {
            input_layernorm_2048<<<mOutside, 256>>>((float *)output_addr, (const float *)input_addr, (const float *)mDeviceGamma, 
                (const float *)mDeviceBeta, mOutside, mInside, mEps, RMSNorm);
        } else if(mInside == 1024) {
            input_layernorm_1024<<<mOutside, 256>>>((float *)output_addr, (const float *)input_addr, (const float *)mDeviceGamma, 
                (const float *)mDeviceBeta, mOutside, mInside, mEps, RMSNorm);
        } else if(mInside == 512) {
            input_layernorm_512<<<mOutside, 256>>>((float *)output_addr, (const float *)input_addr, (const float *)mDeviceGamma, 
                (const float *)mDeviceBeta, mOutside, mInside, mEps, RMSNorm);
        } else if(mInside == 320) {
            input_layernorm_320<<<mOutside, 64>>>((float *)output_addr, (const float *)input_addr, (const float *)mDeviceGamma, 
                (const float *)mDeviceBeta, mOutside, mInside, mEps, RMSNorm);
        } else {
            int sumPerKnl = (mInside+255) / 256;
            input_layernorm<<<mOutside, 256>>>((float *)output_addr, (const float *)input_addr, (const float *)mDeviceGamma, 
                (const float *)mDeviceBeta, mOutside, mInside, mEps, sumPerKnl, RMSNorm);
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
