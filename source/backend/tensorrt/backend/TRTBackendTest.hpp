//
//  TRTBackendTest.hpp
//  MNN
//
//  Created by MNN on 2019/09/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTBackend_TEST_H
#define MNN_TRTBackend_TEST_H
#include <NvInfer.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "TRTType.hpp"
namespace MNN {
const char* kInputTensor  = "input";
const char* kOutputTensor = "output";

class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
        switch (severity) {
            case Severity::kINFO:
                std::cout << msg;
                break;
            case Severity::kWARNING:
                std::cout << msg;
                break;
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                std::cout << msg;
                break;
            default:
                break;
        }
    }
};

class ScopedWeights {
public:
    ScopedWeights(float value) : value_(value) {
        w.type   = nvinfer1::DataType::kFLOAT;
        w.values = &value_;
        w.count  = 1;
    }
    const nvinfer1::Weights& get() {
        return w;
    }

private:
    float value_;
    nvinfer1::Weights w;
};

inline void Execute(IExecutionContext* context, const float* input, float* output) {
    const ICudaEngine& engine = context->getEngine();
    // Two binds, input and output
    MNN_ASSERT(engine.getNbBindings() == 2);
    const int input_index  = engine.getBindingIndex(kInputTensor);
    const int output_index = engine.getBindingIndex(kOutputTensor);
    // Create GPU buffers and a stream
    void* buffers[2]{nullptr, nullptr};
    MNN_ASSERT(0 == cudaMalloc(&buffers[input_index], sizeof(float)));
    MNN_ASSERT(0 == cudaMalloc(&buffers[output_index], sizeof(float)));
    if (buffers[input_index] == nullptr) {
        printf("buffers[input_index] \n");
    }
    cudaStream_t stream;
    MNN_ASSERT(0 == cudaStreamCreate(&stream));
    // Copy the input to the GPU, execute the network, and copy the output back.
    MNN_ASSERT(0 == cudaMemcpyAsync(buffers[input_index], input, sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueue(1, buffers, stream, nullptr);
    MNN_ASSERT(0 == cudaMemcpyAsync(output, buffers[output_index], sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    MNN_ASSERT(0 == cudaFree(buffers[input_index]));
    MNN_ASSERT(0 == cudaFree(buffers[output_index]));
}

// Creates a network to compute y = 2x + 3
inline IHostMemory* CreateNetwork() {
    TRTLogger logger;
    // Create the engine.
    IBuilder* builder = createInferBuilder(logger);
    ScopedWeights weights(2.);
    ScopedWeights bias(3.);

    INetworkDefinition* network = builder->createNetwork();
    // Add the input
    auto input = network->addInput(kInputTensor, nvinfer1::DataType::kFLOAT, DimsCHW{1, 1, 1});
    MNN_ASSERT(input != nullptr);
    // Add the hidden layer.
    auto layer = network->addFullyConnected(*input, 1, weights.get(), bias.get());
    MNN_ASSERT(layer != nullptr);
    // Mark the output.
    auto output = layer->getOutput(0);

    auto inputDims = output->getDimensions();
    printf("nbDims : %d \n", inputDims.nbDims);
    // printf("conv in : \n [");
    // for (size_t i = 0; i < inputDims.nbDims; i++)
    // {
    //     printf("%d, ", inputDims.d[i]);
    // }
    // printf(" ]\n");

    output->setName(kOutputTensor);
    network->markOutput(*output);
    // Build the engine.
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 10);
    auto engine = builder->buildCudaEngine(*network);
    MNN_ASSERT(engine != nullptr);
    // Serialize the engine to create a model, then close.
    IHostMemory* model = engine->serialize();
    network->destroy();
    engine->destroy();
    builder->destroy();
    return model;
}

inline void testTensorRT() {
    printf("\n ====================== testTensorRT Start ====================== \n");
    nvinfer1::IHostMemory* model = CreateNetwork();
    Logger logger;
    nvinfer1::IRuntime* runtime   = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(model->data(), model->size(), nullptr);
    model->destroy();
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // Execute the network.
    float input = 1234;
    float output;
    Execute(context, &input, &output);
    MNN_ASSERT(output == input * 2 + 3);

    // Destroy the engine.
    context->destroy();
    engine->destroy();
    runtime->destroy();
    printf("\n ====================== testTensorRT End ====================== \n");
}
} // namespace MNN
#endif // MNN_TRTBackend_TEST_H
