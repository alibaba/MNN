//
//  BackendTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include "core/Backend.hpp"
#include <MNN/MNNDefine.h>
#include "MNNTestSuite.h"
#include <MNN/Tensor.hpp>

using namespace MNN;

void NCHW2NHWC(const float* source, float* dest, int b, int h, int w, int c) {
    int sourceBatchsize = h * w * c;
    int destBatchSize   = sourceBatchsize;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        for (int hi = 0; hi < h; ++hi) {
            auto srcHeight = srcBatch + hi * w;
            auto dstHeight = dstBatch + hi * w * c;
            for (int wi = 0; wi < w; ++wi) {
                auto srcWidth = srcHeight + wi;
                auto dstWidth = dstHeight + wi * c;
                for (int ci = 0; ci < c; ++ci) {
                    dstWidth[ci] = srcWidth[ci * w * h];
                }
            }
        }
    }
}

void MNNTensorConvertNHWCToNC4HW4(float* dst, const float* src, size_t area, size_t depth) {
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * c;
        float* dstHeight       = dst + hi * 4;
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int i = 0; i < 4; ++i) {
                dstHeight[ci * area * 4 + i] = srcHeight[4 * ci + i];
            }
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + area * cAlign;
    auto dstAlign = dst + area * cAlign;

    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = srcAlign + hi * c;
        float* dstHeight       = dstAlign + hi * 4;

        for (int i = 0; i < 4; ++i) {
            dstHeight[i] = 0;
        }

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNTensorConvertNC4HW4ToNHWC(float* dst, const float* src, size_t area, size_t depth) {
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * 4;
        float* dstHeight       = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int i = 0; i < 4; ++i) {
                dstHeight[ci * 4 + i] = srcHeight[4 * ci * area + i];
            }
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + area * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = srcAlign + hi * 4;
        float* dstHeight       = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void NHWC2NCHW(const float* source, float* dest, int b, int h, int w, int c) {
    int sourceBatchsize = h * w * c;
    int destBatchSize   = sourceBatchsize;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        for (int hi = 0; hi < h; ++hi) {
            auto srcHeight = srcBatch + hi * w * c;
            auto dstHeight = dstBatch + hi * w;
            for (int wi = 0; wi < w; ++wi) {
                auto dstWidth = dstHeight + wi;
                auto srcWidth = srcHeight + wi * c;
                for (int ci = 0; ci < c; ++ci) {
                    dstWidth[ci * w * h] = srcWidth[ci];
                }
            }
        }
    }
}

bool nhwc_2_nhwc_uint8(std::shared_ptr<Backend> bn) {
    MNN_PRINT("\n ========= check NHWC result ! ========= \n");
    std::shared_ptr<Tensor> hostTensor(Tensor::create<uint8_t>(std::vector<int>{1, 224, 224, 3}));
    auto elementSize = hostTensor->elementSize();
    auto hostData    = hostTensor->host<uint8_t>();
    for (int i = 0; i < elementSize; ++i) {
        int flagRandom = i % 255;
        hostData[i]    = flagRandom;
    }

    std::shared_ptr<Tensor> deviceTensor(Tensor::createDevice<uint8_t>(std::vector<int>{1, 224, 224, 3}));
    bn->onAcquireBuffer(deviceTensor.get(), Backend::DYNAMIC_SEPERATE);

    bn->onCopyBuffer(hostTensor.get(), deviceTensor.get());

    std::shared_ptr<Tensor> checkHostTensor(Tensor::create<uint8_t>(std::vector<int>{1, 224, 224, 3}));
    bn->onCopyBuffer(deviceTensor.get(), checkHostTensor.get());

    auto backendCopyData = checkHostTensor->host<uint8_t>();

    for (int i = 0; i < elementSize; ++i) {
        if (backendCopyData[i] != hostData[i]) {
            MNN_PRINT("Error for bn:%d, %d -> %d\n", i, hostData[i], backendCopyData[i]);
            return false;
        }
    }
    return true;
}

bool NC4HW4_2_NC4HW4_float(std::shared_ptr<Backend> bn) {
    MNN_PRINT("\n ========= check NC4HW4 result ! ========= \n");

    std::shared_ptr<Tensor> hostTensor(
        Tensor::create<float>(std::vector<int>{1, 224, 224, 8}, nullptr, Tensor::CAFFE_C4));
    auto elementSize = hostTensor->elementSize();
    auto hostData    = hostTensor->host<float>();
    for (int i = 0; i < elementSize; ++i) {
        int flagRandom = i % 255;
        hostData[i]    = flagRandom;
    }

    std::shared_ptr<Tensor> deviceTensor(Tensor::createDevice<float>(std::vector<int>{1, 224, 224, 8}, Tensor::CAFFE));
    bn->onAcquireBuffer(deviceTensor.get(), Backend::DYNAMIC_SEPERATE);

    bn->onCopyBuffer(hostTensor.get(), deviceTensor.get());

    std::shared_ptr<Tensor> checkHostTensor(
        Tensor::create<float>(std::vector<int>{1, 224, 224, 8}, nullptr, Tensor::CAFFE_C4));
    bn->onCopyBuffer(deviceTensor.get(), checkHostTensor.get());

    auto backendCopyData = checkHostTensor->host<float>();

    for (int i = 0; i < elementSize; ++i) {
        if (backendCopyData[i] != hostData[i]) {
            MNN_PRINT("Error for NCHW Mid bn:%d, %f -> %f\n", i, hostData[i], backendCopyData[i]);
            return false;
        }
    }

    std::shared_ptr<Tensor> deviceTensor2(
        Tensor::createDevice<float>(std::vector<int>{1, 8, 224, 224}, Tensor::TENSORFLOW));
    bn->onAcquireBuffer(deviceTensor2.get(), Backend::DYNAMIC_SEPERATE);
    bn->onCopyBuffer(hostTensor.get(), deviceTensor2.get());
    bn->onCopyBuffer(deviceTensor2.get(), checkHostTensor.get());
    for (int i = 0; i < elementSize; ++i) {
        if (backendCopyData[i] != hostData[i]) {
            MNN_PRINT("Error for NHWC Mid bn:%d, %f -> %f\n", i, hostData[i], backendCopyData[i]);
            return false;
        }
    }
    return true;
}

void NC4HW4_2_NC4HW4_uint8(std::shared_ptr<Backend> bn) {
    MNN_PRINT("\n ========= check NC4HW4 result ! ========= \n");
    std::shared_ptr<Tensor> hostTensor(
        Tensor::create<uint8_t>(std::vector<int>{1, 8, 224, 224}, nullptr, Tensor::CAFFE_C4));
    auto elementSize = hostTensor->elementSize();
    auto hostData    = hostTensor->host<uint8_t>();
    for (int i = 0; i < elementSize; ++i) {
        int flagRandom = i % 255;
        hostData[i]    = flagRandom;
    }

    std::shared_ptr<Tensor> deviceTensor(Tensor::createDevice<uint8_t>(std::vector<int>{1, 224, 224, 8}));
    bn->onAcquireBuffer(deviceTensor.get(), Backend::DYNAMIC_SEPERATE);

    bn->onCopyBuffer(hostTensor.get(), deviceTensor.get());

    std::shared_ptr<Tensor> checkHostTensor(
        Tensor::create<uint8_t>(std::vector<int>{1, 8, 224, 224}, nullptr, Tensor::CAFFE_C4));
    bn->onCopyBuffer(deviceTensor.get(), checkHostTensor.get());

    auto backendCopyData = checkHostTensor->host<uint8_t>();

    for (int i = 0; i < elementSize; ++i) {
        if (backendCopyData[i] != hostData[i]) {
            MNN_PRINT("Error for bn:%d, %d -> %d\n", i, hostData[i], (int32_t)backendCopyData[i]);
            break;
        }
    }
}

void nhwc_2_nhwc_float(std::shared_ptr<Backend> bn) {
    MNN_PRINT("\n ========= check NHWC result ! ========= \n");
    std::shared_ptr<Tensor> hostTensor(Tensor::create<float>(std::vector<int>{1, 224, 224, 3}));
    auto elementSize = hostTensor->elementSize();
    auto hostData    = hostTensor->host<float>();
    for (int i = 0; i < elementSize; ++i) {
        int flagRandom    = (rand() % 2 == 0);
        float valueRandom = rand() % 255 / 255.f;
        hostData[i]       = ((flagRandom == 1) ? 1.0 : -1.0) * valueRandom;
    }

    std::shared_ptr<Tensor> deviceTensor(Tensor::createDevice<float>(std::vector<int>{1, 224, 224, 3}));
    bn->onAcquireBuffer(deviceTensor.get(), Backend::DYNAMIC);

    bn->onCopyBuffer(hostTensor.get(), deviceTensor.get());

    std::shared_ptr<Tensor> checkHostTensor(Tensor::create<float>(std::vector<int>{1, 224, 224, 3}));
    bn->onCopyBuffer(deviceTensor.get(), checkHostTensor.get());

    auto backendCopyData = checkHostTensor->host<float>();

    for (int i = 0; i < elementSize; ++i) {
        if (backendCopyData[i] - hostData[i] >= 0.001f) {
            MNN_PRINT("Error for bn:%d, %f -> %f\n", i, hostData[i], backendCopyData[i]);
        }
    }
}

void nchw_2_nchw_float(std::shared_ptr<Backend> bn) {
    MNN_PRINT("\n ========= check NCHW result ! ========= \n");
    std::shared_ptr<Tensor> hostTensor(Tensor::create<float>(std::vector<int>{1, 7, 224, 224}, nullptr, Tensor::CAFFE));
    auto elementSize = hostTensor->elementSize();
    auto hostData    = hostTensor->host<float>();
    for (int i = 0; i < elementSize; ++i) {
        int flagRandom    = (rand() % 2 == 0);
        float valueRandom = rand() % 255 / 255.f;
        hostData[i]       = ((flagRandom == 1) ? 1.0 : -1.0) * valueRandom;
    }

    std::shared_ptr<Tensor> deviceTensor(Tensor::createDevice<float>(std::vector<int>{1, 224, 224, 7}));
    bn->onAcquireBuffer(deviceTensor.get(), Backend::DYNAMIC_SEPERATE);

    bn->onCopyBuffer(hostTensor.get(), deviceTensor.get());

    std::shared_ptr<Tensor> checkHostTensor(
        Tensor::create<float>(std::vector<int>{1, 7, 224, 224}, nullptr, Tensor::CAFFE));
    bn->onCopyBuffer(deviceTensor.get(), checkHostTensor.get());

    auto backendCopyData = checkHostTensor->host<float>();

    for (int i = 0; i < elementSize; ++i) {
        if (abs(backendCopyData[i] - hostData[i]) >= 0.001f) {
            MNN_PRINT("Error for bn:%d, %f -> %f\n", i, hostData[i], backendCopyData[i]);
        }
    }
}

void nchw_2_NC4HW4_float(std::shared_ptr<Backend> bn) {
    MNN_PRINT("\n ========= check NC4HW4 result ! ========= \n");
    int batch   = 1;
    int channel = 12;
    int width   = 20;
    int height  = 20;
    std::shared_ptr<Tensor> hostTensor(
        Tensor::create<float>(std::vector<int>{batch, channel, height, width}, nullptr, Tensor::CAFFE));
    auto elementSize = hostTensor->elementSize();
    auto hostData    = hostTensor->host<float>();
    for (int i = 0; i < elementSize; ++i) {
        int flagRandom    = (rand() % 2 == 0);
        float valueRandom = rand() % 255 / 255.f;
        hostData[i]       = ((flagRandom == 1) ? 1.0 : -1.0) * valueRandom;
    }

    float* temp = (float*)malloc(hostTensor->size());
    memset(temp, 0.0f, hostTensor->size());
    NCHW2NHWC(hostData, temp, batch, height, width, channel);

    std::shared_ptr<Tensor> deviceTensor(Tensor::createDevice<float>(std::vector<int>{batch, height, width, channel}));
    bn->onAcquireBuffer(deviceTensor.get(), Backend::DYNAMIC_SEPERATE);
    bn->onCopyBuffer(hostTensor.get(), deviceTensor.get());

    //            // nhwc -> NC4HW4
    //            MNN_PRINT("nhwc -> NC4HW4 !\n");

    MNNTensorConvertNHWCToNC4HW4(hostData, temp, height * width, channel);
    std::shared_ptr<Tensor> NC4HW4_HostTensor(
        Tensor::create<float>(std::vector<int>{batch, channel, height, width}, nullptr, Tensor::CAFFE_C4));

    bn->onCopyBuffer(deviceTensor.get(), NC4HW4_HostTensor.get());
    auto backendCopyData = NC4HW4_HostTensor->host<float>();

    for (int i = 0; i < elementSize; ++i) {
        if (abs(backendCopyData[i] - hostData[i]) >= 0.001f) {
            MNN_PRINT("Error for bn:%d, %f -> %f\n", i, hostData[i], backendCopyData[i]);
        }
    }

    // NC4HW4 -> nhwc

    MNNTensorConvertNC4HW4ToNHWC(temp, hostData, height * width, channel);

    bn->onCopyBuffer(NC4HW4_HostTensor.get(), deviceTensor.get());
    NHWC2NCHW(temp, backendCopyData, batch, height, width, channel);
    bn->onCopyBuffer(deviceTensor.get(), hostTensor.get());

    //            MNN_PRINT("NC4HW4 -> nhwc !\n");
    for (int i = 0; i < elementSize; ++i) {
        if (abs(backendCopyData[i] - hostData[i]) >= 0.001f) {
            MNN_PRINT("Error for bn:%d, %f -> %f\n", i, hostData[i], backendCopyData[i]);
        }
    }

    free(temp);
}

void nchw_2_NC4HW4_2_nchw_float(std::shared_ptr<Backend> bn) {
    // Test NCHW -> NC4HW4 -> NCHW
    {
        std::shared_ptr<Tensor> hostTensor(
            Tensor::create<float>(std::vector<int>{1, 3, 224, 224}, nullptr, Tensor::CAFFE));
        auto elementSize = hostTensor->elementSize();
        auto hostData    = hostTensor->host<float>();
        for (int i = 0; i < elementSize; ++i) {
            hostData[i] = ((i * 67 * 73) % 255);
        }

        std::shared_ptr<Tensor> deviceTensor(
            Tensor::createDevice<float>(std::vector<int>{1, 3, 224, 224}, Tensor::CAFFE_C4));
        bn->onAcquireBuffer(deviceTensor.get(), Backend::DYNAMIC);

        bn->onCopyBuffer(hostTensor.get(), deviceTensor.get());
        std::shared_ptr<Tensor> checkHostTensor(
            Tensor::create<float>(std::vector<int>{1, 3, 224, 224}, nullptr, Tensor::CAFFE));
        bn->onCopyBuffer(deviceTensor.get(), checkHostTensor.get());

        auto backendCopyData = checkHostTensor->host<float>();

        for (int i = 0; i < elementSize; ++i) {
            if (backendCopyData[i] != hostData[i]) {
                MNN_PRINT("Error for bn:%d, %f -> %f\n", i, hostData[i], backendCopyData[i]);
                break;
            }
        }
    }
}

void nhwc_2_NC4HW4_2_nhwc_float(std::shared_ptr<Backend> bn) {
    // Test NHWC -> NC4HW4 -> NHWC
    MNN_PRINT("\n ========= check NC4HW4 result ! ========= \n");
    int batch   = 1;
    int channel = 12;
    int width   = 20;
    int height  = 20;
    std::shared_ptr<Tensor> hostTensor(
        Tensor::create<float>(std::vector<int>{batch, channel, height, width}, nullptr, Tensor::CAFFE));
    auto elementSize = hostTensor->elementSize();
    auto hostData    = hostTensor->host<float>();
    for (int i = 0; i < elementSize; ++i) {
        int flagRandom    = (rand() % 2 == 0);
        float valueRandom = rand() % 255 / 255.f;
        hostData[i]       = ((flagRandom == 1) ? 1.0 : -1.0) * valueRandom;
    }

    float* temp = (float*)malloc(hostTensor->size());
    memset(temp, 0.0f, hostTensor->size());
    NCHW2NHWC(hostData, temp, batch, height, width, channel);

    std::shared_ptr<Tensor> deviceTensor(Tensor::createDevice<float>(std::vector<int>{batch, height, width, channel}));
    bn->onAcquireBuffer(deviceTensor.get(), Backend::DYNAMIC_SEPERATE);
    bn->onCopyBuffer(hostTensor.get(), deviceTensor.get());

    //            // nhwc -> NC4HW4
    //            MNN_PRINT("nhwc -> NC4HW4 !\n");

    MNNTensorConvertNHWCToNC4HW4(hostData, temp, height * width, channel);
    std::shared_ptr<Tensor> NC4HW4_HostTensor(
        Tensor::create<float>(std::vector<int>{batch, channel, height, width}, nullptr, Tensor::CAFFE_C4));

    bn->onCopyBuffer(deviceTensor.get(), NC4HW4_HostTensor.get());
    auto backendCopyData = NC4HW4_HostTensor->host<float>();

    for (int i = 0; i < elementSize; ++i) {
        if (abs(backendCopyData[i] - hostData[i]) >= 0.001f) {
            MNN_PRINT("Error for bn:%d, %f -> %f\n", i, hostData[i], backendCopyData[i]);
        }
    }

    // NC4HW4 -> nhwc

    MNNTensorConvertNC4HW4ToNHWC(temp, hostData, height * width, channel);

    bn->onCopyBuffer(NC4HW4_HostTensor.get(), deviceTensor.get());
    NHWC2NCHW(temp, backendCopyData, batch, height, width, channel);
    bn->onCopyBuffer(deviceTensor.get(), hostTensor.get());

    //            MNN_PRINT("NC4HW4 -> nhwc !\n");
    for (int i = 0; i < elementSize; ++i) {
        if (abs(backendCopyData[i] - hostData[i]) >= 0.001f) {
            MNN_PRINT("Error for bn:%d, %f -> %f\n", i, hostData[i], backendCopyData[i]);
        }
    }

    free(temp);
}

class BackendCopyBufferTest : public MNNTestCase {
public:
    virtual bool run();
    virtual ~BackendCopyBufferTest() = default;
};

bool BackendCopyBufferTest::run() {
    for (int i = 0; i < MNN_FORWARD_ALL; ++i) {
        auto type    = (MNNForwardType)i;
        auto creator = MNNGetExtraBackendCreator(type);
        if (nullptr == creator)
            continue;

        MNN_PRINT("Test %d Backend\n", type);
        MNN::Backend::Info info;
        info.type = type;
        std::shared_ptr<Backend> bn(creator->onCreate(info));

        // uint8
        auto res = nhwc_2_nhwc_uint8(bn);
        res = res && NC4HW4_2_NC4HW4_float(bn);
        if (!res) {
            MNN_ERROR("Error for %d bn\n", i);
            return false;
        }
        //        NC4HW4_2_NC4HW4_uint8(bn);
    }
    return true;
}

MNNTestSuiteRegister(BackendCopyBufferTest, "engine/backend/copy_buffer");
