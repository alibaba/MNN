//
//  BackendTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include <math.h>
#include <MNN/Tensor.hpp>
#include "MNNTestSuite.h"
#include "core/Backend.hpp"
#include "core/Macro.h"

using namespace MNN;

template <typename T>
void NCHW2NHWC(const T* source, T* dest, int b, int h, int w, int c) {
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

template <typename T>
void MNNTensorConvertNHWCToNC4HW4(T* dst, const T* src, size_t area, size_t depth) {
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        const auto srcHeight = src + hi * c;
        auto dstHeight       = dst + hi * 4;
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
        const auto srcHeight = srcAlign + hi * c;
        auto dstHeight       = dstAlign + hi * 4;

        for (int i = 0; i < 4; ++i) {
            dstHeight[i] = 0;
        }

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

template <typename T>
void MNNTensorConvertNC4HW4ToNHWC(T* dst, const T* src, size_t area, size_t depth) {
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        const auto srcHeight = src + hi * 4;
        auto dstHeight       = dst + hi * c;
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
        const auto srcHeight = srcAlign + hi * 4;
        auto dstHeight       = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

template <typename T>
void NHWC2NCHW(const T* source, T* dest, int b, int h, int w, int c) {
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
        if (abs(backendCopyData[i] - hostData[i]) >= F32_BF16_MAX_LOSS) {
            MNN_PRINT("Error for bn:%d, %d -> %d\n", i, hostData[i], backendCopyData[i]);
            return false;
        }
    }
    return true;
}

template <typename T>
bool NC4HW4_2_NC4HW4_IntType(std::shared_ptr<Backend> bn) {
    MNN_PRINT("\n ========= check NC4HW4_2_NC4HW4_IntType result ! ========= \n");

    std::shared_ptr<Tensor> hostTensor(
        Tensor::create<T>(std::vector<int>{1, 224, 224, 8}, nullptr, Tensor::CAFFE_C4));
    auto elementSize = hostTensor->elementSize();
    auto hostData    = hostTensor->host<T>();
    for (int i = 0; i < elementSize; ++i) {
        int flagRandom = i % 255;
        hostData[i]    = flagRandom;
    }

    bn->onResizeBegin();
    std::shared_ptr<Tensor> deviceTensor_pre(Tensor::createDevice<T>(std::vector<int>{1, 224, 224, 8}, Tensor::CAFFE_C4));
    bn->onAcquireBuffer(deviceTensor_pre.get(), Backend::STATIC);
    std::shared_ptr<Tensor> deviceTensor(Tensor::createDevice<T>(std::vector<int>{1, 224, 224, 8}, Tensor::CAFFE_C4));
    bn->onAcquireBuffer(deviceTensor.get(), Backend::STATIC);
    bn->onCopyBuffer(hostTensor.get(), deviceTensor_pre.get());
    bn->onCopyBuffer(deviceTensor_pre.get(), deviceTensor.get());

    std::shared_ptr<Tensor> checkHostTensor(
        Tensor::create<T>(std::vector<int>{1, 224, 224, 8}, nullptr, Tensor::CAFFE_C4));
    bn->onCopyBuffer(deviceTensor.get(), checkHostTensor.get());

    auto backendCopyData = checkHostTensor->host<T>();

    for (int i = 0; i < elementSize; ++i) {
        if (abs(backendCopyData[i] - hostData[i]) >= F32_BF16_MAX_LOSS) {
            MNN_PRINT("Error for NCHW Mid bn:%d, %d -> %d\n", i, hostData[i], backendCopyData[i]);
            return false;
        }
    }

    std::shared_ptr<Tensor> deviceTensor2(
        Tensor::createDevice<T>(std::vector<int>{1, 8, 224, 224}, Tensor::TENSORFLOW));
    bn->onAcquireBuffer(deviceTensor2.get(), Backend::DYNAMIC_SEPERATE);
    bn->onReleaseBuffer(deviceTensor2.get(), Backend::DYNAMIC_SEPERATE);
    bn->onResizeEnd();
    bn->onCopyBuffer(hostTensor.get(), deviceTensor2.get());
    bn->onCopyBuffer(deviceTensor2.get(), checkHostTensor.get());
    for (int i = 0; i < elementSize; ++i) {
        if (abs(backendCopyData[i] - hostData[i]) >= F32_BF16_MAX_LOSS) {
            MNN_PRINT("Error for NHWC Mid bn:%d, %d -> %d\n", i, hostData[i], backendCopyData[i]);
            return false;
        }
    }
    return true;
}

bool NCHW_NC4HW4_NCHW(std::shared_ptr<Backend> bn, int batch, int width, int height, int channel) {
    std::shared_ptr<Tensor> srcTensor(
        Tensor::create<float>({batch, channel, width, height}, nullptr, Tensor::CAFFE));
    auto host = srcTensor->host<float>();
    for (int b=0; b<batch; ++b) {
        for (int c=0; c<channel; ++c) {
            for (int y=0; y<height; ++y) {
                for (int x=0; x<width; ++x) {
                    host[0
                         + b * channel * height * width
                         + c * height * width
                         + y * width
                         + x
                         ] = b / (float)batch * 100.f + c / (float)channel * 10.f + y / (float)height * 0.1f + x / (float)width * 0.001f;
                }
            }
        }
    }
    std::shared_ptr<Tensor> dstTensor(
        Tensor::create<float>({batch, channel, width, height}, nullptr, Tensor::CAFFE));
    std::shared_ptr<Tensor> deviceTensor(Tensor::createDevice<float>({batch, channel, width, height}, Tensor::CAFFE_C4));
    bn->onAcquireBuffer(deviceTensor.get(), Backend::STATIC);
    bn->onCopyBuffer(srcTensor.get(), deviceTensor.get());
    bn->onCopyBuffer(deviceTensor.get(), dstTensor.get());
    int elementSize = srcTensor->elementSize();
    auto backendCopyData = dstTensor->host<float>();
    auto hostData = srcTensor->host<float>();
    for (int i = 0; i < elementSize; ++i) {
        if (abs(backendCopyData[i] - hostData[i]) >= 0.1f) {
            MNN_PRINT("Error for bn:%d, %f -> %f, %f\n", i, hostData[i], backendCopyData[i], F32_BF16_MAX_LOSS);
            return false;
        }
    }
    bn->onReleaseBuffer(deviceTensor.get(), Backend::STATIC);
    return true;
}

bool NC4HW4_2_NC4HW4_float(std::shared_ptr<Backend> bn) {
//    MNN_PRINT("\n ========= check NC4HW4_2_NC4HW4_float result ! ========= \n");
    std::vector<int> nhwc_shape = {1, 32, 12, 13};
    std::vector<int> nchw_shape = {1, 12, 13, 32};
    std::shared_ptr<Tensor> hostTensor(
        Tensor::create<float>(nhwc_shape, nullptr, Tensor::CAFFE_C4));
    auto elementSize = hostTensor->elementSize();
    auto hostData    = hostTensor->host<float>();
    for (int i = 0; i < elementSize; ++i) {
        int flagRandom = i % 255;
        hostData[i]    = flagRandom;
    }

    bn->onResizeBegin();
//    MNN_PRINT("\nalloc deviceTensor_pre\n");
    std::shared_ptr<Tensor> deviceTensor_pre(Tensor::createDevice<float>(nhwc_shape, Tensor::CAFFE_C4));
    bn->onAcquireBuffer(deviceTensor_pre.get(), Backend::STATIC);

//    MNN_PRINT("\nalloc deviceTensor");
    std::shared_ptr<Tensor> deviceTensor(Tensor::createDevice<float>(nhwc_shape, Tensor::CAFFE_C4));
    bn->onAcquireBuffer(deviceTensor.get(), Backend::STATIC);

//    MNN_PRINT("\ncopy from host to  deviceTensor_pre\n");
    bn->onCopyBuffer(hostTensor.get(), deviceTensor_pre.get());

//    MNN_PRINT("\ncopy from deviceTensor_pre to  deviceTensor\n");
    bn->onCopyBuffer(deviceTensor_pre.get(), deviceTensor.get());

//    MNN_PRINT("\ncopy from deviceTensor to  new host\n");
    std::shared_ptr<Tensor> checkHostTensor(
        Tensor::create<float>(nhwc_shape, nullptr, Tensor::CAFFE_C4));
    bn->onCopyBuffer(deviceTensor.get(), checkHostTensor.get());


    auto backendCopyData = checkHostTensor->host<float>();

    for (int i = 0; i < elementSize; ++i) {
        if (backendCopyData[i] != hostData[i]) {
            MNN_PRINT("Error for NCHW Mid bn:%d, %f -> %f\n", i, hostData[i], backendCopyData[i]);
            return false;
        }
    }

    std::shared_ptr<Tensor> deviceTensor2(
        Tensor::createDevice<float>(nchw_shape, Tensor::TENSORFLOW));
    bn->onAcquireBuffer(deviceTensor2.get(), Backend::DYNAMIC_SEPERATE);
    bn->onReleaseBuffer(deviceTensor2.get(), Backend::DYNAMIC_SEPERATE);
    bn->onResizeEnd();
    bn->onCopyBuffer(hostTensor.get(), deviceTensor2.get());
    bn->onCopyBuffer(deviceTensor2.get(), checkHostTensor.get());
    for (int i = 0; i < elementSize; ++i) {
        if (abs(backendCopyData[i] - hostData[i]) >= F32_BF16_MAX_LOSS) {
            MNN_PRINT("Error for NHWC Mid bn:%d, %f -> %f\n", i, hostData[i], backendCopyData[i]);
            return false;
        }
    }
    return true;
}

void NC4HW4_2_NC4HW4_uint8(std::shared_ptr<Backend> bn) {
//    MNN_PRINT("\n ========= check NC4HW4 result ! ========= \n");
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
        if (abs(backendCopyData[i] - hostData[i]) >= F32_BF16_MAX_LOSS) {
            MNN_PRINT("Error for bn:%d, %d -> %d\n", i, hostData[i], (int32_t)backendCopyData[i]);
            break;
        }
    }
}

void nhwc_2_nhwc_float(std::shared_ptr<Backend> bn) {
//    MNN_PRINT("\n ========= check NHWC result ! ========= \n");
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
        if (backendCopyData[i] - hostData[i] >= F32_BF16_MAX_LOSS) {
            MNN_PRINT("Error for bn:%d, %f -> %f\n", i, hostData[i], backendCopyData[i]);
        }
    }
}

void nchw_2_nchw_float(std::shared_ptr<Backend> bn) {
//    MNN_PRINT("\n ========= check NCHW result ! ========= \n");
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
        if (abs(backendCopyData[i] - hostData[i]) >= F32_BF16_MAX_LOSS) {
            MNN_PRINT("Error for bn:%d, %f -> %f\n", i, hostData[i], backendCopyData[i]);
        }
    }
}

void nchw_2_NC4HW4_float(std::shared_ptr<Backend> bn) {
//    MNN_PRINT("\n ========= check NC4HW4 result ! ========= \n");
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
        if (abs(backendCopyData[i] - hostData[i]) >= F32_BF16_MAX_LOSS) {
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
        if (abs(backendCopyData[i] - hostData[i]) >= F32_BF16_MAX_LOSS) {
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
            if (abs(backendCopyData[i] != hostData[i]) >= F32_BF16_MAX_LOSS) {
                MNN_PRINT("Error for bn:%d, %f -> %f\n", i, hostData[i], backendCopyData[i]);
                break;
            }
        }
    }
}

template <typename T>
bool nhwc_2_NC4HW4_2_nhwc_inttype(std::shared_ptr<Backend> bn) {
    // Test NHWC -> NC4HW4 -> NHWC
    MNN_PRINT("\n ========= check nhwc_2_NC4HW4_2_nhwc_inttype result ! ========= \n");
    int batch   = 1;
    int channel = 12;
    int width   = 20;
    int height  = 20;
    std::shared_ptr<Tensor> hostTensor(
        Tensor::create<T>(std::vector<int>{batch, channel, height, width}, nullptr, Tensor::CAFFE));
    auto elementSize = hostTensor->elementSize();
    auto hostData    = hostTensor->host<T>();
    for (int i = 0; i < elementSize; ++i) {
        hostData[i]       = rand() % 255;
    }

    T* temp = (T*)malloc(hostTensor->size());
    memset(temp, 0.0f, hostTensor->size());
    NCHW2NHWC<T>(hostData, temp, batch, height, width, channel);

    std::shared_ptr<Tensor> deviceTensor_pre(Tensor::createDevice<T>(std::vector<int>{batch, height, width, channel}));
    bn->onAcquireBuffer(deviceTensor_pre.get(), Backend::STATIC);
    std::shared_ptr<Tensor> deviceTensor(Tensor::createDevice<T>(std::vector<int>{batch, height, width, channel}));
    bn->onAcquireBuffer(deviceTensor.get(), Backend::STATIC);
    bn->onCopyBuffer(hostTensor.get(), deviceTensor_pre.get());
    bn->onCopyBuffer(deviceTensor_pre.get(), deviceTensor.get());

    //            // nhwc -> NC4HW4
    //            MNN_PRINT("nhwc -> NC4HW4 !\n");

    MNNTensorConvertNHWCToNC4HW4<T>(hostData, temp, height * width, channel);
    std::shared_ptr<Tensor> NC4HW4_HostTensor(
        Tensor::create<T>(std::vector<int>{batch, channel, height, width}, nullptr, Tensor::CAFFE_C4));

    bn->onCopyBuffer(deviceTensor.get(), NC4HW4_HostTensor.get());
    auto backendCopyData = NC4HW4_HostTensor->host<T>();

    for (int i = 0; i < elementSize; ++i) {
        if (abs(backendCopyData[i] - hostData[i]) >= F32_BF16_MAX_LOSS) {
            MNN_PRINT("Error for bn:%d, %d -> %d\n", i, hostData[i], backendCopyData[i]);
            return false;
        }
    }

    // NC4HW4 -> nhwc

    MNNTensorConvertNC4HW4ToNHWC<T>(temp, hostData, height * width, channel);

    bn->onCopyBuffer(NC4HW4_HostTensor.get(), deviceTensor.get());
    NHWC2NCHW(temp, backendCopyData, batch, height, width, channel);
    bn->onCopyBuffer(deviceTensor.get(), hostTensor.get());

    // MNN_PRINT("NC4HW4 -> nhwc !\n");
    for (int i = 0; i < elementSize; ++i) {
        if (abs(backendCopyData[i] - hostData[i]) >= F32_BF16_MAX_LOSS) {
            MNN_PRINT("Error for bn:%d, %d -> %d\n", i, hostData[i], backendCopyData[i]);
        }
    }

    free(temp);
    return true;
}
bool nchwTonhwc(std::shared_ptr<Backend> bn) {
    // Test NHWC -> NC4HW4 -> NHWC
    MNN_PRINT("\n ========= check nchwTonhwc result ! ========= \n");
    int batch   = 2;
    int channel = 12;
    int width   = 21;
    int height  = 5;
    std::shared_ptr<Tensor> hostTensor(
        Tensor::create<float>(std::vector<int>{batch, channel, height, width}, nullptr, Tensor::CAFFE));
    auto elementSize = hostTensor->elementSize();
    auto hostData    = hostTensor->host<float>();
    for (int i = 0; i < elementSize; ++i) {
        int flagRandom    = (rand() % 2 == 0);
        float valueRandom = rand() % 255 / 255.f;
        hostData[i]       = ((flagRandom == 1) ? 1.0 : -1.0) * valueRandom;
    }
    std::vector<float> tempStorage(hostTensor->elementSize());
    float* temp = tempStorage.data();
    memset(temp, 0.0f, hostTensor->size());
    NCHW2NHWC(hostData, temp, batch, height, width, channel);
    std::shared_ptr<Tensor> deviceTensor_pre(Tensor::createDevice<float>(std::vector<int>{batch, height, width, channel}));
    bn->onAcquireBuffer(deviceTensor_pre.get(), Backend::STATIC);
    std::shared_ptr<Tensor> deviceTensor(Tensor::createDevice<float>(std::vector<int>{batch, height, width, channel}));
    bn->onAcquireBuffer(deviceTensor.get(), Backend::STATIC);
    bn->onCopyBuffer(hostTensor.get(), deviceTensor_pre.get());
    bn->onCopyBuffer(deviceTensor_pre.get(), deviceTensor.get());
    std::shared_ptr<Tensor> hostTensorNHWC(
        Tensor::create<float>(std::vector<int>{batch, height, width, channel}, nullptr, Tensor::TENSORFLOW));
    bn->onCopyBuffer(deviceTensor.get(), hostTensorNHWC.get());
    auto backendCopyData = hostTensorNHWC->host<float>();
    for (int i = 0; i < elementSize; ++i) {
        if (abs(backendCopyData[i] - temp[i]) >= F32_BF16_MAX_LOSS) { //Error of converting from float32 to bf16 is more than 0.001
            MNN_PRINT("Error for bn:%d, %f -> %f. F32_BF16_MAX_LOSS:%f\n", i, temp[i], backendCopyData[i], F32_BF16_MAX_LOSS);
            return false;
        }
    }
    return true;
}


bool nhwc_2_NC4HW4_2_nhwc_float(std::shared_ptr<Backend> bn) {
    // Test NHWC -> NC4HW4 -> NHWC
    MNN_PRINT("\n ========= check nhwc_2_NC4HW4_2_nhwc_float result ! ========= \n");
    int batch   = 1;
    int channel = 12;
    int width   = 3;
    int height  = 2;
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
    bn->onAcquireBuffer(deviceTensor.get(), Backend::STATIC);
    bn->onCopyBuffer(hostTensor.get(), deviceTensor.get());

    // // nhwc -> NC4HW4
    // MNN_PRINT("nhwc -> NC4HW4 !\n");

    MNNTensorConvertNHWCToNC4HW4(hostData, temp, height * width, channel);
    std::shared_ptr<Tensor> NC4HW4_HostTensor(
        Tensor::create<float>(std::vector<int>{batch, channel, height, width}, nullptr, Tensor::CAFFE_C4));

    bn->onCopyBuffer(deviceTensor.get(), NC4HW4_HostTensor.get());
    auto backendCopyData = NC4HW4_HostTensor->host<float>();

    bool res = true;
    for (int i = 0; i < elementSize; ++i) {
        if (abs(backendCopyData[i] - hostData[i]) >= F32_BF16_MAX_LOSS) { //Error of converting from float32 to bf16 is more than 0.001
            MNN_PRINT("Error for bn:%d, %f -> %f. F32_BF16_MAX_LOSS:%f\n", i, hostData[i], backendCopyData[i], F32_BF16_MAX_LOSS);
            res = false;
            break;
        }
    }
    if (!res) {
        for (int i = 0; i < elementSize; ++i) {
            MNN_PRINT("%d, %f -> %f. F32_BF16_MAX_LOSS:%f\n", i, hostData[i], backendCopyData[i], F32_BF16_MAX_LOSS);
        }
        return false;
    }

    // NC4HW4 -> nhwc

    MNNTensorConvertNC4HW4ToNHWC(temp, hostData, height * width, channel);

    bn->onCopyBuffer(NC4HW4_HostTensor.get(), deviceTensor.get());
    NHWC2NCHW(temp, backendCopyData, batch, height, width, channel);
    bn->onCopyBuffer(deviceTensor.get(), hostTensor.get());

    MNN_PRINT("NC4HW4 -> nhwc !\n");
    for (int i = 0; i < elementSize; ++i) {
        if (abs(backendCopyData[i] - hostData[i]) >= F32_BF16_MAX_LOSS) {
            MNN_PRINT("NC4HW4 -> nhwc Error for bn:%d, %f -> %f.  F32_BF16_MAX_LOSS:%f\n", i, hostData[i], backendCopyData[i], F32_BF16_MAX_LOSS);
            return false;
        }
    }

    free(temp);
    return true;
}

class BackendCopyBufferFloatTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        for (int i = 0; i < MNN_FORWARD_ALL; ++i) {
            auto type    = (MNNForwardType)i;
            auto creator = MNNGetExtraRuntimeCreator(type);
            if (nullptr == creator) {
                continue;
            }
            for (int p = 0; p < 3; ++p) {
                MNN::Backend::Info info;
                info.type = type;
                BackendConfig user;
                user.precision = (MNN::BackendConfig::PrecisionMode)p;
                info.user = &user;
                std::shared_ptr<Runtime> runtime(creator->onCreate(info));
                MNN_PRINT("Test %d Backend for %d \n", type, user.precision);
                std::shared_ptr<Backend> bn(runtime->onCreate(&user));
                auto res = NC4HW4_2_NC4HW4_float(bn);
                FUNC_PRINT(res);
                res = res && nchwTonhwc(bn);
                FUNC_PRINT(res);
                res = res && nhwc_2_NC4HW4_2_nhwc_float(bn);
                FUNC_PRINT(res);
                res = res && NCHW_NC4HW4_NCHW(bn, 3, 16, 17, 19);
                FUNC_PRINT(res);
                res = res && NCHW_NC4HW4_NCHW(bn, 12, 16, 38, 16);
                FUNC_PRINT(res);
                res = res && NCHW_NC4HW4_NCHW(bn, 5, 128, 8, 6);
                FUNC_PRINT(res);
                if (!res) {
                    MNN_ERROR("Error for %d bn\n", i);
                    return false;
                }
            }
        }
        return true;
    }
};

class CPUBackendCopyBufferTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto type    = MNN_FORWARD_CPU;
        auto creator = MNNGetExtraRuntimeCreator(type);
        for (int p = 0; p < 3; ++p) {
            MNN::Backend::Info info;
            info.type = type;
            BackendConfig user;
            user.precision = (MNN::BackendConfig::PrecisionMode)p;
            info.user = &user;
            std::shared_ptr<Runtime> runtime(creator->onCreate(info));
            MNN_PRINT("Test %d Backend for %d \n", type, user.precision);
            std::shared_ptr<Backend> bn(runtime->onCreate(&user));
            auto res = NC4HW4_2_NC4HW4_IntType<int32_t>(bn);
            res = res && NC4HW4_2_NC4HW4_IntType<int16_t>(bn);
            res = res && NC4HW4_2_NC4HW4_IntType<int8_t>(bn);
            res = res && nhwc_2_NC4HW4_2_nhwc_inttype<int32_t>(bn);
            res = res && nhwc_2_NC4HW4_2_nhwc_inttype<int16_t>(bn);
            res = res && nhwc_2_NC4HW4_2_nhwc_inttype<int8_t>(bn);
            if (!res) {
                MNN_ERROR("Error for Int Copy\n");
                return false;
            }
        }
        return true;
    }
};

class BackendCopyBufferUint8Test : public MNNTestCase {
public:
    virtual bool run(int precision) {
        for (int i = 0; i < MNN_FORWARD_ALL; ++i) {
            auto type    = (MNNForwardType)i;
            auto creator = MNNGetExtraRuntimeCreator(type);
            if (nullptr == creator) {
                continue;
            }
            MNN::Backend::Info info;
            info.type = type;
            BackendConfig user;
            user.precision = MNN::BackendConfig::Precision_High;
            info.user = &user;
            std::shared_ptr<Runtime> runtime(creator->onCreate(info));

            MNN_PRINT("Test %d Backend\n", type);
            std::shared_ptr<Backend> bn(runtime->onCreate());
            // uint8
            auto res = nhwc_2_nhwc_uint8(bn);
            if (!res) {
                MNN_ERROR("Error for %d bn\n", i);
                return false;
            }
            //        NC4HW4_2_NC4HW4_uint8(bn);
        }
        return true;
    }
};
MNNTestSuiteRegister(BackendCopyBufferFloatTest, "engine/backend/copy_buffer_float");
//MNNTestSuiteRegister(BackendCopyBufferUint8Test, "engine/backend/copy_buffer_uint8");
MNNTestSuiteRegister(CPUBackendCopyBufferTest, "engine/backend/copy_buffer_cpu");
