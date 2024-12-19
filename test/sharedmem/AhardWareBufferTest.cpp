//
//  ReplaceTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __ANDROID__
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include <MNN/expr/Module.hpp>
#include "TestUtils.h"
#include <android/hardware_buffer.h>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

using namespace MNN;
using namespace MNN::Express;

static AHardwareBuffer* creatAHardwareBuffer(int width, int height, void *data){
    // 创建和初始化硬件缓冲区
    AHardwareBuffer_Desc bufferDesc = {};
    bufferDesc.width = width;
    bufferDesc.height = height;
    bufferDesc.layers = 1;
    bufferDesc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
    bufferDesc.usage = AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE;

    AHardwareBuffer* buffer = nullptr;
    int result = AHardwareBuffer_allocate(&bufferDesc, &buffer);
    if(result != 0) {
        MNN_ERROR("alloc AHardwareBuffer failed   %d\n", result);
    }
    
    if(nullptr != data){
        void* map = nullptr;
        ARect rect = { 0, 0, width, height };  // Define the region to lock
        result = AHardwareBuffer_lock(buffer, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, &rect, &map);
        if (result != 0) {
            MNN_ERROR("Handle lock failed\n");
        }
        if (map) {
            memcpy(map, data, width * height * 4);
        }
        
        AHardwareBuffer_unlock(buffer, nullptr);
    }
    return buffer;
}

static void ReleaseAHardWareBuffer(AHardwareBuffer* buffer){
    AHardwareBuffer_release(buffer);
}

static void copyDataFromAHardWareBuffer(AHardwareBuffer* buffer, int width, int height, void *data){
    int result = 0;
    if(nullptr != data){
        void* map = nullptr;
        ARect rect = { 0, 0, width, height };  // Define the region to lock
        result = AHardwareBuffer_lock(buffer, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, &rect, &map);
        if (result != 0) {
            MNN_ERROR("Handle lock failed\n");
        }
        if (map) {
            memcpy(data, map, width * height * 4);
        }
        
        AHardwareBuffer_unlock(buffer, nullptr);
    }
}

static bool checkvalue(const float* ref, const unsigned char* out, int size){
    for(int i = 0; i < size; ++i){
        if(ref[i] != (float)out[i]){
            MNN_ERROR("%d:  ref %f != out %f\n", i, ref[i], (float)out[i]);
            return false;
        }
    }
    return true;
}

const int width = 1280;
const int height = 720;
const int channel = 3;
static std::shared_ptr<Module> _createModel() {
    auto x = _Input({1, channel, height, width}, NCHW, halide_type_of<float>());
    x->setName("Input");
    auto y = _Transpose(x, {0, 1, 3, 2});
    y->setName("Transpose");
    std::unique_ptr<NetT> net(new NetT);
    Variable::save({y}, net.get());
    flatbuffers::FlatBufferBuilder builder;
    auto len = MNN::Net::Pack(builder, net.get());
    builder.Finish(len);
    return std::shared_ptr<Module>(Module::load({"Input"}, {"Transpose"}, builder.GetBufferPointer(), builder.GetSize()));
}
// Test prepareCompute for dynamic-graph usage
class AhardWareBufferTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        if (MNN_FORWARD_OPENCL != getCurrentType()) {
            MNN_ERROR("Currently forwardtype[%d] run sharedmem/AhardWareBuffer has error, skip it\n", getCurrentType());
            return true;
        }
        auto net = _createModel();
        auto x = _Input({1, channel, height, width}, NCHW, halide_type_of<float>());
        unsigned char inputData[4 * height * width];
        unsigned char outputData[4 * height * width];
        for(int i = 0; i < 4 * height * width; ++i){
            inputData[i] = i;
        }
        // ahardwarebuffer default format is nc4hw4
        {
            auto xPtr = x->writeMap<float>();
            for (int i = 0; i < channel; ++i){
                for (int j = 0; j < height * width; ++j) {
                    xPtr[i * height * width + j] = (float)inputData[j * 4 + i];
                }
            }
            x->unMap();
        }
        
        auto outputs = net->onForward({x});
        outputs[0] = _Convert(outputs[0], NC4HW4);
        auto refPtr = outputs[0]->readMap<float>();
        auto size = outputs[0]->getInfo()->size;
        
        auto xShared = _Input({1, channel, height, width}, NCHW, halide_type_of<float>());
        auto inputAhardwareBuffer = creatAHardwareBuffer(width, height, inputData);
        volatile uint64_t inputValue = (uint64_t)inputAhardwareBuffer;
        xShared->setDevicePtr((void*)inputValue, MNN_MEMORY_AHARDWAREBUFFER);
        auto outputsShared = net->onForward({xShared});
        auto outputAhardwareBuffer = creatAHardwareBuffer(width, height, nullptr);
        volatile uint64_t outputValue = (uint64_t)inputAhardwareBuffer;
        {
            outputsShared[0]->copyToDevicePtr((void*)outputValue, MNN_MEMORY_AHARDWAREBUFFER);
            copyDataFromAHardWareBuffer(inputAhardwareBuffer, width, height, outputData);
            if(checkvalue(refPtr, outputData, size) == false){
                MNN_ERROR("sharedmem/AhardWareBuffer test failed!\n");
                return false;
            }
        }
        
        // speed
        const auto time = 100;
        {
            MNN::Timer _t;
            for (int t = 0; t < time; ++t) {
                x->writeMap<float>();
                auto outputs = net->onForward({x});
                outputs[0]->readMap<float>();
            }
            float timeCost = _t.durationInUs() / 1000.0f / (float)time;
            MNN_PRINT("cpu copy [%d, %d, %d], Avg time: %f ms\n", channel, height, width, timeCost);
        }
        {
            MNN::Timer _t;
            for (int t = 0; t < time; ++t) {
                xShared->setDevicePtr((void*)inputValue, MNN_MEMORY_AHARDWAREBUFFER);
                auto outputs = net->onForward({xShared});
                outputs[0]->copyToDevicePtr((void*)outputValue, MNN_MEMORY_AHARDWAREBUFFER);
            }
            float timeCost = _t.durationInUs() / 1000.0f / (float)time;
            MNN_PRINT("shared memory copy [%d, %d, %d], Avg time: %f ms\n", channel, height, width, timeCost);
        }
        
        ReleaseAHardWareBuffer(inputAhardwareBuffer);
        ReleaseAHardWareBuffer(outputAhardwareBuffer);
        return true;
    }
};

MNNTestSuiteRegister(AhardWareBufferTest, "sharedmem/AhardWareBuffer");
#endif
