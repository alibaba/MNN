//
//  Tensor.cpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Tensor.hpp"
#include <complex.h>
#include <string.h>
#include "Backend.hpp"
#include "MNNMemoryUtils.h"
#include "MNN_generated.h"
#include "Macro.h"
#include "TensorUtils.hpp"

#define MAX_TENSOR_DIM 6

using namespace std;

namespace MNN {
Tensor::Tensor(int dimSize, DimensionType type) {
    MNN_ASSERT(dimSize <= MAX_TENSOR_DIM);

    mBuffer.dim        = new halide_dimension_t[MAX_TENSOR_DIM];
    mBuffer.dimensions = dimSize;
    mBuffer.type       = halide_type_of<float>();
    mBuffer.device     = 0;
    mBuffer.host       = nullptr;

    mDescribe                   = new InsideDescribe;
    mDescribe->dimensionStorage = mBuffer.dim;
    switch (type) {
        case CAFFE:
            mDescribe->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            break;
        case TENSORFLOW:
            mDescribe->dimensionFormat = MNN_DATA_FORMAT_NHWC;
            break;
        case CAFFE_C4:
            mDescribe->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            mBuffer.dim[1].flags       = REORDER_4;
            break;
        default:
            break;
    }
}

Tensor::Tensor(const Tensor* tensor, DimensionType type, bool allocMemory) {
    MNN_ASSERT(tensor != nullptr);

    auto buffer        = tensor->buffer();
    mBuffer.dim        = new halide_dimension_t[MAX_TENSOR_DIM];
    mBuffer.dimensions = buffer.dimensions;
    mBuffer.type       = buffer.type;
    mBuffer.device     = 0;
    mBuffer.host       = nullptr;
    for (int i = 0; i < buffer.dimensions; ++i) {
        mBuffer.dim[i].min    = 0;
        mBuffer.dim[i].extent = buffer.dim[i].extent;
    }
    mDescribe                   = new InsideDescribe;
    mDescribe->dimensionStorage = mBuffer.dim;
    switch (type) {
        case CAFFE:
            mDescribe->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            break;
        case TENSORFLOW:
            mDescribe->dimensionFormat = MNN_DATA_FORMAT_NHWC;
            break;
        case CAFFE_C4:
            mDescribe->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            mBuffer.dim[1].flags       = REORDER_4;
            type                       = CAFFE;
            break;
        default:
            break;
    }

    // format mapping
    auto originType = tensor->getDimensionType();
    if (4 == buffer.dimensions && originType != type) {
        std::vector<int> axisMap;
        // NCHW -> NHWC
        if (originType == CAFFE) {
            axisMap = std::vector<int>{0, 2, 3, 1};
        }
        // NHWC -> NCHW
        else {
            axisMap = std::vector<int>{0, 3, 1, 2};
        }
        for (int i = 0; i < buffer.dimensions; ++i) {
            mBuffer.dim[i].extent = buffer.dim[axisMap[i]].extent;
        }
    }
    TensorUtils::setLinearLayout(this);

    if (allocMemory) {
        mDescribe->ownHost = true;
        mBuffer.host       = (uint8_t*)MNNMemoryAllocAlign(size(), MNN_MEMORY_ALIGN_DEFAULT);
        MNN_ASSERT(mBuffer.host != nullptr);
    }
}

Tensor::~Tensor() {
    if (nullptr != mDescribe->handleFreeFunction) {
        MNN_ASSERT(mBuffer.type.code == halide_type_handle);
        auto handles = (void**)mBuffer.host;
        for (int i = 0; i < elementSize(); ++i) {
            if (nullptr != handles[i]) {
                mDescribe->handleFreeFunction(handles[i]);
            }
        }
    }
    if (mDescribe->ownHost) {
        MNNMemoryFreeAlign(mBuffer.host);
    }
    delete[] mDescribe->dimensionStorage;
    delete mDescribe;
}

Tensor* Tensor::createDevice(const std::vector<int>& dims, halide_type_t type, DimensionType dimType) {
    Tensor shapeTensor((int)dims.size(), dimType);
    for (int i = 0; i < dims.size(); ++i) {
        shapeTensor.setLength(i, dims[i]);
    }
    shapeTensor.buffer().type = type;
    return new Tensor(&shapeTensor, dimType, false);
}

Tensor* Tensor::create(const std::vector<int>& dims, halide_type_t type, void* userData, DimensionType dimType) {
    Tensor shapeTensor((int)dims.size(), dimType);
    for (int i = 0; i < dims.size(); ++i) {
        shapeTensor.setLength(i, dims[i]);
    }
    shapeTensor.buffer().type = type;

    bool ownData = userData == nullptr;
    auto result  = new Tensor(&shapeTensor, dimType, ownData);
    if (nullptr != userData) {
        result->buffer().host = (uint8_t*)userData;
    }
    return result;
}

bool Tensor::copyFromHostTensor(const Tensor* hostTensor) {
    auto bn = mDescribe->backend;
    if (nullptr == bn) {
        return false;
    }
    bn->onCopyBuffer(hostTensor, this);
    return true;
}

bool Tensor::copyToHostTensor(Tensor* hostTensor) const {
    auto bn = mDescribe->backend;
    if (nullptr == bn) {
        return false;
    }
    bn->onCopyBuffer(this, hostTensor);
    return true;
}

static Tensor::DimensionType getDimType(const Tensor* origin) {
    auto dimformat = TensorUtils::getDescribe(origin)->dimensionFormat;
    switch (dimformat) {
        case MNN_DATA_FORMAT_NHWC:
            return Tensor::TENSORFLOW;
        case MNN_DATA_FORMAT_NCHW:
            return Tensor::CAFFE;
        case MNN_DATA_FORMAT_NC4HW4:
            return Tensor::CAFFE_C4;
        default:
            break;
    }
    return Tensor::CAFFE;
}

Tensor* Tensor::createHostTensorFromDevice(const Tensor* device, bool copyContent) {
    auto tensor = Tensor::create(device->shape(), device->getType(), nullptr, getDimType(device));
    if (copyContent) {
        device->copyToHostTensor(tensor);
    }
    return tensor;
}

Tensor::DimensionType Tensor::getDimensionType() const {
    if (mDescribe->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
        return Tensor::TENSORFLOW;
    }
    return Tensor::CAFFE;
}

Tensor::HandleDataType Tensor::getHandleDataType() const {
    if (halide_type_handle != mBuffer.type.code) {
        return HANDLE_NONE;
    }
    return mDescribe->handleType;
}

void Tensor::setType(int type) {
    switch (type) {
        case DataType_DT_DOUBLE:
        case DataType_DT_FLOAT:
            mBuffer.type = halide_type_of<float>();
            break;
        case DataType_DT_BFLOAT16:
            mBuffer.type = halide_type_t(halide_type_float, 16);
            break;
        case DataType_DT_QINT32:
        case DataType_DT_INT32:
        case DataType_DT_BOOL:
        case DataType_DT_INT64:
            mBuffer.type = halide_type_of<int32_t>();
            break;
        case DataType_DT_QINT8:
        case DataType_DT_INT8:
            mBuffer.type = halide_type_of<int8_t>();
            break;
        case DataType_DT_QUINT8:
        case DataType_DT_UINT8:
            mBuffer.type = halide_type_of<uint8_t>();
            break;
        case DataType_DT_QUINT16:
        case DataType_DT_UINT16:
            mBuffer.type = halide_type_of<uint16_t>();
            break;
        case DataType_DT_QINT16:
        case DataType_DT_INT16:
            mBuffer.type = halide_type_of<int16_t>();
            break;
        case DataType_DT_STRING:
            mBuffer.type                  = halide_type_t(halide_type_handle, sizeof(void*) * 8);
            mDescribe->handleType         = HANDLE_STRING;
            mDescribe->handleFreeFunction = (void (*)(void*))::free;
            break;

        default:
            MNN_PRINT("Unsupported data type!");
            MNN_ASSERT(false);
            break;
    }
}

std::vector<int> Tensor::shape() const {
    std::vector<int> result;
    for (int i = 0; i < mBuffer.dimensions; ++i) {
        result.push_back(mBuffer.dim[i].extent);
    }
    return result;
}

int Tensor::size() const {
    auto dataSize = this->buffer().type.bytes();
    MNN_ASSERT(dataSize >= 1);
    for (int i = 0; i < this->buffer().dimensions; i++) {
        int currentDimSize = mBuffer.dim[i].extent;
        switch (mBuffer.dim[i].flags) {
            case REORDER_4:
                currentDimSize = ALIGN_UP4(currentDimSize);
                break;
            case REORDER_8:
                currentDimSize = ALIGN_UP8(currentDimSize);
                break;
            default:
                break;
        }
        dataSize *= currentDimSize;
    }
    return dataSize;
}

template <typename T>
void printData(const Tensor* tensor, const void* data, const char* fmt) {
    const T* buffer = (const T*)data;
    if (tensor->dimensions() != 4) {
        for (int i = 0; i < tensor->elementSize(); i++) {
            printf(fmt, buffer[i]);
        }
        MNN_PRINT("\n");
        return;
    }

    auto tf      = tensor->getDimensionType() == Tensor::TENSORFLOW;
    auto batch   = tensor->batch();
    auto channel = tensor->channel();
    auto height  = tensor->height();
    auto width   = tensor->width();

    auto unit = sizeof(T);
    if (tf) {
        auto bytesPerRow   = channel * unit;
        auto bytesPerImage = width * bytesPerRow;
        auto bytesPerBatch = height * bytesPerImage;

        for (int b = 0; b < batch; b++) {
            auto bytes = buffer + b * bytesPerBatch / unit;
            MNN_PRINT("batch %d:\n", b);

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    for (int c = 0; c < channel; c++) {
                        printf(fmt, bytes[h * width * channel + w * channel + c]);
                    }
                    MNN_PRINT("\n");
                }
                MNN_PRINT("--------------\n");
            }
        }
    } else if (tensor->buffer().dim[1].flags == Tensor::REORDER_4) { // NC/4HW4
        auto components    = 4;
        auto bytesPerRow   = width * components * unit;
        auto bytesPerImage = height * bytesPerRow;
        auto bytesPerBatch = UP_DIV(channel, 4) * bytesPerImage;

        for (int b = 0; b < batch; b++) {
            auto bytes = buffer + b * bytesPerBatch / unit;
            MNN_PRINT("batch %d:\n", b);

            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        auto n = c / components, r = c % components;
                        printf(fmt, bytes[(n * width * height + h * width + w) * components + r]);
                    }
                    MNN_PRINT("\n");
                }
                MNN_PRINT("--------------\n");
            }
        }
    } else { // NCHW
        auto bytesPerRow   = width * unit;
        auto bytesPerImage = height * bytesPerRow;
        auto bytesPerBatch = channel * bytesPerImage;

        for (int b = 0; b < batch; b++) {
            auto bytes = buffer + b * bytesPerBatch / unit;
            MNN_PRINT("batch %d:\n", b);

            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        printf(fmt, bytes[c * width * height + h * width + w]);
                    }
                    MNN_PRINT("\n");
                }
                MNN_PRINT("--------------\n");
            }
        }
    }
}

void Tensor::print() const {
    // print dimensions
    MNN_PRINT("====== Tensor %p ======", this);
    MNN_PRINT("\nDimension: ");
    for (int i = 0; i < mBuffer.dimensions; i++) {
        MNN_PRINT("%d, ", mBuffer.dim[i].extent);
    }

    // convert to host if needed
    auto printee = this;
    bool device  = this->buffer().host == NULL && this->buffer().device != 0;
    if (device) {
        printee = this->createHostTensorFromDevice(this, true);
    }
    auto buffer = printee->buffer().host;

    MNN_PRINT("\nData: ");
    if (printee->getType().code == halide_type_int) {
        if (printee->getType().bits == 8) { // int8
            printData<int8_t>(printee, buffer, "%d, ");
        } else if (printee->getType().bits == 16) { // int16
            printData<int16_t>(printee, buffer, "%d, ");
        } else if (printee->getType().bits == 32) { // int32
            printData<int32_t>(printee, buffer, "%d, ");
        } else if (printee->getType().bits == 64) { // int64
            printData<int64_t>(printee, buffer, "%ld, ");
        } else {
            MNN_PRINT("\nunsupported data type");
        }
    } else if (printee->getType().code == halide_type_uint) {
        if (printee->getType().bits == 8) { // uint8
            printData<uint8_t>(printee, buffer, "%d, ");
        } else if (printee->getType().bits == 16) { // uint16
            printData<uint16_t>(printee, buffer, "%d, ");
        } else if (printee->getType().bits == 32) { // uint32
            printData<uint32_t>(printee, buffer, "%d, ");
        } else if (printee->getType().bits == 64) { // uint64
            printData<uint64_t>(printee, buffer, "%ld, ");
        } else {
            MNN_PRINT("\nunsupported data type");
        }
    } else if (printee->getType().code == halide_type_float) {
        if (printee->getType().bits == 32) { // float32
            printData<float>(printee, buffer, "%f, ");
        }
#ifdef __FLT16_EPSILON__
        else if (printee->getType().bits == 16) { // float16
            printData<__fp16>(printee, buffer, "%f, ");
        }
#endif
        else {
            MNN_PRINT("\nunsupported data type");
        }
    } else {
        MNN_PRINT("\nunsupported data type");
    }

    // clean up
    if (printee != this) {
        delete printee;
    }
}

} // namespace MNN
