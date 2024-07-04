//
//  Tensor.cpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <complex.h>
#include <string.h>
#include <MNN/Tensor.hpp>
#include "core/Backend.hpp"
#include "core/MNNMemoryUtils.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

using namespace std;

namespace MNN {
Tensor::Tensor(int dimSize, DimensionType type) {
    MNN_ASSERT(dimSize <= MNN_MAX_TENSOR_DIM);
    mDescribe          = new InsideDescribe;
    mDescribe->mContent.reset(new InsideDescribe::NativeInsideDescribe);
    auto nativeDescribe = mDescribe->mContent.get();
    mBuffer.dimensions = dimSize;
    mBuffer.type       = halide_type_of<float>();
    mBuffer.device     = 0;
    mBuffer.host       = nullptr;
    mBuffer.dim        = &nativeDescribe->dims[0];
    mBuffer.flags      = 0;

    switch (type) {
        case CAFFE:
            nativeDescribe->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            break;
        case TENSORFLOW:
            nativeDescribe->dimensionFormat = MNN_DATA_FORMAT_NHWC;
            break;
        case CAFFE_C4:
            nativeDescribe->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            break;
        default:
            break;
    }
}

Tensor::Tensor(const Tensor* tensor, DimensionType type, bool allocMemory) {
    MNN_ASSERT(tensor != nullptr);

    auto buffer        = tensor->buffer();
    mDescribe          = new InsideDescribe;
    mDescribe->mContent.reset(new InsideDescribe::NativeInsideDescribe);
    auto nativeDescribe = mDescribe->mContent.get();
    mBuffer.dimensions = buffer.dimensions;
    mBuffer.type       = buffer.type;
    mBuffer.device     = 0;
    mBuffer.host       = nullptr;
    mBuffer.dim        = &nativeDescribe->dims[0];
    mBuffer.flags      = 0;
    for (int i = 0; i < buffer.dimensions; ++i) {
        mBuffer.dim[i].extent = buffer.dim[i].extent;
    }
    switch (type) {
        case CAFFE:
            nativeDescribe->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            break;
        case TENSORFLOW:
            nativeDescribe->dimensionFormat = MNN_DATA_FORMAT_NHWC;
            break;
        case CAFFE_C4:
            nativeDescribe->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            type                       = CAFFE;
            break;
        default:
            break;
    }

    // format mapping
    auto originType = tensor->getDimensionType();
    if (originType != type && buffer.dimensions >= 4) {
        std::vector<int> axisMap;
        // NCHW -> NHWC
        if (originType == CAFFE) {
            axisMap.push_back(0);
            for (int i = 2; i < buffer.dimensions; ++i) {
                axisMap.push_back(i);
            }
            axisMap.push_back(1);
        }
        // NHWC -> NCHW
        else {
            axisMap.push_back(0);
            axisMap.push_back(buffer.dimensions - 1);
            for (int i = 1; i < buffer.dimensions - 1; ++i) {
                axisMap.push_back(i);
            }
        }
        for (int i = 0; i < buffer.dimensions; ++i) {
            mBuffer.dim[i].extent = buffer.dim[axisMap[i]].extent;
        }
    }
    TensorUtils::setLinearLayout(this);

    for (int i = mBuffer.dimensions; i < 4; i++) {
        mBuffer.dim[i].extent = 1;
    }

    if (allocMemory) {
        auto memorySize = size();
        if (memorySize > 0) {
            nativeDescribe->memoryType = Tensor::InsideDescribe::MEMORY_HOST;
            mBuffer.host          = (uint8_t*)MNNMemoryAllocAlign(size(), MNN_MEMORY_ALIGN_DEFAULT);
            MNN_ASSERT(mBuffer.host != nullptr);
        }
    }
}
Tensor::Tensor(bool deepCopy, const Tensor* tensor) {
    mDescribe = new InsideDescribe;
    mDescribe->mContent = tensor->mDescribe->mContent;
    mDescribe->setBackend(tensor->mDescribe->getBackend());
    mDescribe->mem = tensor->mDescribe->mem;
    mBuffer.dim = TensorUtils::getDescribe(tensor)->dims;
    mBuffer.type = tensor->getType();
    mBuffer.device = tensor->deviceId();
    mBuffer.host = tensor->buffer().host;
    mBuffer.dimensions = tensor->buffer().dimensions;
    mBuffer.flags = tensor->buffer().flags;
}

Tensor::~Tensor() {
    // MNN_PRINT("free tensor:%p\n", this);
    auto nativeDescribe = mDescribe->mContent.get();
    if (nativeDescribe->memoryType == InsideDescribe::MEMORY_HOST) {
        if (nullptr != mBuffer.host) {
            MNNMemoryFreeAlign(mBuffer.host);
        }
    }
    delete mDescribe;
}

Tensor* Tensor::createDevice(const std::vector<int>& dims, halide_type_t type, DimensionType dimType) {
    auto shapeTensor = new Tensor((int)dims.size(), dimType);
    for (int i = 0; i < dims.size(); ++i) {
        shapeTensor->setLength(i, dims[i]);
    }
    shapeTensor->buffer().type = type;
    TensorUtils::setLinearLayout(shapeTensor);
    return shapeTensor;
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
Tensor* Tensor::clone(const Tensor* src, bool deepCopy) {
    return new Tensor(deepCopy, src);
}


bool Tensor::copyFromHostTensor(const Tensor* hostTensor) {
    auto bn = mDescribe->getBackend();
    if (nullptr == bn) {
        return false;
    }
    bn->onCopyBuffer(hostTensor, this);
    return true;
}

bool Tensor::copyToHostTensor(Tensor* hostTensor) const {
    auto bn = mDescribe->getBackend();
    if (nullptr == bn) {
        return false;
    }
    bn->onCopyBuffer(this, hostTensor);
    return true;
}

Tensor* Tensor::createHostTensorFromDevice(const Tensor* device, bool copyContent) {
    auto tensor = Tensor::create(device->shape(), device->getType(), nullptr, TensorUtils::getDimType(device));
    if (copyContent) {
        device->copyToHostTensor(tensor);
    }
    return tensor;
}

Tensor::DimensionType Tensor::getDimensionType() const {
    auto nativeDescribe = mDescribe->mContent.get();
    if (nativeDescribe->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
        return Tensor::TENSORFLOW;
    }
    return Tensor::CAFFE;
}

Tensor::HandleDataType Tensor::getHandleDataType() const {
    if (halide_type_handle != mBuffer.type.code) {
        return HANDLE_NONE;
    }
    return HANDLE_STRING;
}
void Tensor::setType(int type) {
    auto nativeDescribe = mDescribe->mContent.get();
    switch (type) {
        case DataType_DT_DOUBLE:
        case DataType_DT_FLOAT:
            mBuffer.type = halide_type_of<float>();
            break;
        case DataType_DT_BFLOAT16:
            mBuffer.type = halide_type_t(halide_type_bfloat, 16);
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
        default:
            MNN_PRINT("Unsupported data type! %d\n", type);
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
template <typename T>
void printData(const Tensor* tensor, const void* data, const char* fmt) {
    const T* buffer = (const T*)data;
    if (tensor->dimensions() != 4) {
        auto size = tensor->elementSize();
        for (int i = 0; i < size; i++) {
            MNN_PRINT(fmt, buffer[i]);
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
                        MNN_PRINT(fmt, bytes[h * width * channel + w * channel + c]);
                    }
                    MNN_PRINT("\n");
                }
                MNN_PRINT("--------------\n");
            }
        }
    } else if (TensorUtils::getDescribe(tensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) { // NC/4HW4
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
                        MNN_PRINT(fmt, bytes[(n * width * height + h * width + w) * components + r]);
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
                        MNN_PRINT(fmt, bytes[c * width * height + h * width + w]);
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
        } else {
            MNN_PRINT("\nunsupported data type");
        }
    } else if (printee->getType().code == halide_type_uint) {
        if (printee->getType().bits == 8) { // uint8
            printData<uint8_t>(printee, buffer, "%d, ");
        } else {
            MNN_PRINT("\nunsupported data type");
        }
    } else if (printee->getType().code == halide_type_float) {
        if (printee->getType().bits == 32) { // float32
            printData<float>(printee, buffer, "%f, ");
        } else {
            MNN_PRINT("\nunsupported data type\n");
        }
    } else {
        MNN_PRINT("\nunsupported data type");
    }

    // clean up
    if (printee != this) {
        delete printee;
    }
}

void Tensor::printShape() const {
    const int dims = this->dimensions();
    MNN_PRINT("\t**Tensor shape**: ");
    if (dims == 0) {
        MNN_PRINT("\t*Scalar*");
    }
    for (int i = 0; i < dims; ++i) {
        MNN_PRINT("%d, ", this->length(i));
    }
    MNN_PRINT("\n");
}

size_t Tensor::usize() const {
    size_t dataSize = mBuffer.type.bytes();
    MNN_ASSERT(dataSize >= 1);
    auto nativeDescribe = mDescribe->mContent.get();
    for (int i = 0; i < this->buffer().dimensions; i++) {
        int currentDimSize = mBuffer.dim[i].extent;
        if (nativeDescribe->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && 1 == i) {
            currentDimSize = ALIGN_UP4(currentDimSize);
        }
        dataSize *= currentDimSize;
    }
    return dataSize;
}

int Tensor::size() const {
    return static_cast<int>(usize());
}

void* Tensor::map(MapType mtype, DimensionType dtype) {
    auto nativeDescribe = mDescribe;
    auto bn = nativeDescribe->getBackend();
    if (nullptr == bn) {
        return mBuffer.host;
    }

    auto mapPtr = bn->onMapTensor(mtype, dtype, this);
    if(mapPtr != nullptr) {
        // Get mapPtr in specific backend
        return mapPtr;
    }

    /* Common backend */
    auto needSize = this->size();
    void* hostPtr = malloc(needSize);

    if(mtype == Tensor::MAP_TENSOR_READ) {
        //tmpTensor alloc
        MNN::Tensor tmpTensor(this, dtype, false);
        tmpTensor.buffer().host = (uint8_t *)hostPtr;

        //use onCopyBuffer
        bn->onCopyBuffer(this, &tmpTensor);
    }
    return hostPtr;
}

void Tensor::unmap(MapType mtype, DimensionType dtype, void *mapPtr) {
    auto nativeDescribe = mDescribe;
    auto bn = nativeDescribe->getBackend();
    if (nullptr == bn) {
        return;
    }

    bool ret = bn->onUnmapTensor(mtype, dtype, this, mapPtr);
    if(true == ret) {
        //do unmap already, just return
        return;
    }

    if(mtype == Tensor::MAP_TENSOR_WRITE) {
        //srcTensor alloc
        MNN::Tensor srcTensor(this, dtype, false);
        srcTensor.buffer().host = (uint8_t *)mapPtr;

        //use onCopyBuffer
        bn->onCopyBuffer(&srcTensor, this);
    }
    if(mapPtr != nullptr) {
        free(mapPtr);
        mapPtr = nullptr;
    }
}
int Tensor::wait(MapType mtype, bool finish) {
    auto nativeDescribe = mDescribe;
    auto bn = nativeDescribe->getBackend();
    if (nullptr == bn) {
        return 0;
    }
    return bn->onSync(mtype, finish, this);
}

bool Tensor::setDevicePtr(const void* devicePtr, int memoryType) {
    mBuffer.flags = memoryType;
    mBuffer.device = (uint64_t)devicePtr;
    // To use memoryType afterwards
    return true;
}

void Tensor::destroy(Tensor* tensor) {
    if (nullptr != tensor) {
        delete tensor;
    }
}
bool Tensor::getDeviceInfo(void* dst, int type) const {
    auto nativeDescribe = mDescribe;
    if (nullptr == nativeDescribe->getBackend()) {
        return false;
    }
    if (nativeDescribe->getBackend()->type() != type) {
        return false;
    }
    return nativeDescribe->getBackend()->onGetTensorInfo(this, dst);
}

} // namespace MNN
