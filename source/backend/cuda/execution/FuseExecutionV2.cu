#ifdef MNN_CODEGEN_CUDA
#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "backend/cuda/core/compiler/CUDACompiler.hpp"
#include "FuseExecutionV2.hpp"
namespace MNN {
namespace CUDA {

bool FuseExecutionV2::check(const Op* op) {
    if (op->type() != OpType_Extra) {
        return false;
    }
    if (nullptr == op->main_as_Extra()) {
        return false;
    }
    auto extra = op->main_as_Extra();
    if (nullptr == extra->attr()) {
        return false;
    }
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "version") {
            if (nullptr != attr->s()) {
                std::string cont = attr->s()->str();
                return cont == "common";
            }
            return false;
        }
    }
    return false;
}

class FuseExecutionCommon : public Execution {
public:
    FuseExecutionCommon(const Extra* extra, Backend* bn, int inputSize, int outputSize) : Execution(bn) {
        auto cuBn = static_cast<CUDABackend*>(bn);
        mOutputBinding.resize(outputSize);
        mInputBinding.resize(inputSize);
        mGroupSize.resize(3);
        mLocalSize.resize(3);
        // Find shader
        std::pair<string, string> code;
        std::vector<const char*> compile_params;
        code.first = extra->type()->str();
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "cuda") {
                code.second = attr->s()->str();
                break;
            }
        }
        cuBn->compile(&mCuModule, code, compile_params);
        MNN_CUDA_SAFE_CALL(cuModuleGetFunction(&mKernel, mCuModule, code.first.c_str()));

        // Get group size and local size
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "group_size") {
                auto ptr = attr->tensor()->int32s()->data();
                mGroupSize[0] = ptr[0];
                mGroupSize[1] = ptr[1];
                mGroupSize[2] = ptr[2];
                break;
            }
        }
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "local_size") {
                auto ptr = attr->tensor()->int32s()->data();
                mLocalSize[0] = ptr[0];
                mLocalSize[1] = ptr[1];
                mLocalSize[2] = ptr[2];
                break;
            }
        }
        int maxIndex = -1;
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "input") {
                maxIndex = ALIMAX(maxIndex, attr->i());
            } else if (attr->key()->str() == "const") {
                maxIndex = ALIMAX(maxIndex, attr->i());
            }
        }
        mArgs.resize(maxIndex+1);
        auto pool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
        auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "input") {
                auto list = attr->list()->i()->data();
                if (0 == list[0]) {
                    mInputBinding[list[1]] = attr->i();
                } else {
                    mOutputBinding[list[1]] = attr->i();
                }
                continue;
            }
            if (attr->key()->str() == "const") {
                auto b = attr->tensor();
                void* result = nullptr;
                size_t bufferSize = 0;
                switch (b->dataType()) {
                    case DataType_DT_FLOAT:
                        result = (void*)b->float32s()->Data();
                        bufferSize = b->float32s()->size() * sizeof(float);
                        break;
                    case DataType_DT_INT32:
                        result = (void*)b->int32s()->Data();
                        bufferSize = b->int32s()->size() * sizeof(int32_t);
                        break;
                    default:
                        MNN_ASSERT(false);
                        break;
                }
                MemChunk buffer = pool->alloc(bufferSize);
                runtime->memcpy(buffer.ptr(), result, bufferSize, MNNMemcpyHostToDevice);

                mConstIndides.emplace_back(std::make_pair(attr->i(), buffer));
                continue;
            }
        }
    }
    virtual ~ FuseExecutionCommon() {
        auto pool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
        for (auto& iter : mConstIndides) {
            pool->free(iter.second);
        }
        cuModuleUnload(mCuModule);
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return NO_ERROR;
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        for (int i=0; i<mInputBinding.size(); ++i) {
            mArgs[mInputBinding[i]] = (void *)inputs[i]->buffer().device;
        }
        for (int i=0; i<mOutputBinding.size(); ++i) {
            mArgs[mOutputBinding[i]] = (void *)outputs[i]->buffer().device;
        }
        for (auto& iter : mConstIndides) {
            mArgs[iter.first] = iter.second.ptr();
        }
        std::vector<void*> argsPtr;
        for(int i=0; i<mArgs.size(); i++) {
            argsPtr.emplace_back(mArgs.data() + i);
        }

        MNN_CUDA_SAFE_CALL(
            cuLaunchKernel(mKernel,
            mGroupSize[0], mGroupSize[1], mGroupSize[2], // grid dim
            mLocalSize[0], mLocalSize[1], mLocalSize[2], // block dim
            0, NULL, // shared mem 
            &(argsPtr[0]), 0)); // arguments

        return NO_ERROR;
    }


private:
    CUmodule mCuModule;
    CUfunction mKernel;
    std::vector<void*> mArgs;
    std::vector<int> mGroupSize;
    std::vector<int> mLocalSize;
    std::vector<int> mInputBinding;
    std::vector<int> mOutputBinding;
    std::vector<std::pair<int, MemChunk>> mConstIndides;
};

Execution* FuseExecutionV2::create(const Op* op, Backend *backend, int inputSize, int outputSize) {
    return new FuseExecutionCommon(op->main_as_Extra(), backend, inputSize, outputSize);
}


}
};


#endif