#include "BinaryExecution.hpp"
#include "Raster.cuh"
namespace MNN {
namespace CUDA {

template <typename T>
__global__ void ATAN2(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        T x = input0[i * s0];
        T y = input1[i * s1];
        output[i] = atan2(x, y);
    }
    return;
}

template <typename T>
__global__ void MOD(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        T x = input0[i * s0];
        T y = input1[i * s1];
        output[i] = x - x / y;
    }
    return;
}

template <typename T>
__global__ void LOGICALOR(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        T x = input0[i * s0];
        T y = input1[i * s1];
        output[i] = (x || y) ? 1 : 0;
    }
    return;
}
BinaryExecution::BinaryExecution(int opType, Backend *backend, int activationType) : Execution(backend) {
    mType = opType;
    mActivationType = activationType;
}
BinaryExecution::~BinaryExecution(){
    // Do nothing
}
ErrorCode BinaryExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto count = CUDABackend::realSize(outputs[0]);
    auto inputS0 = CUDABackend::realSize(inputs[0]);
    auto inputS1 = CUDABackend::realSize(inputs[1]);
    int s0 = inputS0 == 1 ? 0 : 1;
    int s1 = inputS1 == 1 ? 0 : 1;
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    //printf("%d - %d\n", block_num, threads_num);
    int size[3] = {1, 1, count};
    int stride0[3] = {0, 0, s0};
    int stride1[3] = {0, 0, s1};
    int stride2[3] = {0, 0, 1};

    // use input type. output type maybe fixed, for example greater/less
    auto type = inputs[0]->getType();
    if (type.code == halide_type_float) {
        // Use Half or float
        type.bits = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]) * 8;
    }

    auto computeFunction = [&](Tensor* input0T, Tensor* input1T, Tensor* outputT) {
        auto input0 = (uint8_t*)input0T->deviceId();
        auto input1 = (uint8_t*)input1T->deviceId();
        auto output = (uint8_t*)outputT->deviceId();
        BinaryBlit(output, input0, input1, size, stride0, stride1, stride2, type, runtime, mType, mActivationType);
    };
    computeFunction(inputs[0], inputs[1], outputs[0]);
    for (int i=2; i<inputs.size(); ++i) {
        computeFunction(outputs[0], inputs[i], outputs[0]);
    }
    return NO_ERROR;
}
class BinaryCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (op->type() == OpType_BinaryOp) {
        #ifdef ENABLE_CUDA_QUANT
            if (CUDABackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
                return new BinaryInt8Execution(op, backend);
            }
        #endif
            // MNN_PRINT("binary act:%d %d\n", op->main_as_BinaryOp()->opType(), op->main_as_BinaryOp()->activationType());
            return new BinaryExecution(op->main_as_BinaryOp()->opType(), backend, op->main_as_BinaryOp()->activationType());
        }
        if (op->type() == OpType_Eltwise) {
            switch (op->main_as_Eltwise()->type()) {
                case EltwiseType_SUM:
                    return new BinaryExecution(BinaryOpOperation_ADD, backend);
                case EltwiseType_PROD:
                    return new BinaryExecution(BinaryOpOperation_MUL, backend);
                case EltwiseType_MAXIMUM:
                    return new BinaryExecution(BinaryOpOperation_MAXIMUM, backend);
                default:
                    break;
            }
        }
        return nullptr;
    }
};

static CUDACreatorRegister<BinaryCreator> __init(OpType_BinaryOp);
static CUDACreatorRegister<BinaryCreator> __init2(OpType_Eltwise);
}
}
