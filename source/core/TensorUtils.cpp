//
//  TensorUtils.cpp
//  MNN
//
//  Created by MNN on 2018/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TensorUtils.hpp"
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <cmath>
#include <cstring>
#include "Backend.hpp"
#include "Macro.h"

namespace MNN {
Tensor::InsideDescribe* TensorUtils::getDescribe(const Tensor* tensor) {
    return tensor->mDescribe;
}

void TensorUtils::copyShape(const Tensor* source, Tensor* dest, bool copyFormat) {
    auto& ob      = dest->buffer();
    auto& ib      = source->buffer();
    ob.dimensions = ib.dimensions;
    ::memcpy(ob.dim, ib.dim, ib.dimensions * sizeof(halide_dimension_t));
    if (copyFormat) {
        getDescribe(dest)->dimensionFormat = getDescribe(source)->dimensionFormat;
    }
}

void TensorUtils::setLinearLayout(Tensor* tensor) {
    auto& buffer = tensor->buffer();
    int size     = 1;
    for (int i = 0; i < buffer.dimensions; ++i) {
        auto index  = buffer.dimensions - i - 1;
        auto extent = buffer.dim[index].extent;
        switch (buffer.dim[index].flags) {
            case Tensor::REORDER_4:
                extent = ROUND_UP(extent, 4);
                break;
            case Tensor::REORDER_8:
                extent = ROUND_UP(extent, 8);
                break;
            default:
                break;
        }

        buffer.dim[index].stride = size;
        size *= extent;
    }
}

void TensorUtils::clearHandleData(Tensor* tensor) {
    if (tensor->buffer().type.code != halide_type_handle) {
        return;
    }
    auto handle = tensor->host<void*>();
    if (nullptr == handle) {
        return;
    }

    MNN_ASSERT(tensor->mDescribe->handleFreeFunction != nullptr);
    for (int i = 0; i < tensor->elementSize(); ++i) {
        if (nullptr != handle[i]) {
            tensor->mDescribe->handleFreeFunction(handle[i]);
            handle[i] = nullptr;
        }
    }
}

static const Tensor* createHostPlanar(const Tensor* source) {
    // check
    bool device = source->buffer().host == NULL && source->buffer().device != 0;
    bool chunky = false;
    for (int i = 0; i < source->dimensions(); i++) {
        if (source->buffer().dim[i].flags) {
            chunky = true;
            break;
        }
    }

    // no convert needed
    if (!device && !chunky) {
        return source;
    }

    // convert
    if (chunky) {
        Tensor* result = source->createHostTensorFromDevice(source, false);
        if (result->getDimensionType() == MNN::Tensor::TENSORFLOW) {
            TensorUtils::getDescribe(result)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        } else {
            TensorUtils::getDescribe(result)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        }
        for (int i = 0; i < source->dimensions(); i++) {
            result->buffer().dim[i].flags = 0;
        }
        TensorUtils::setLinearLayout(result);

        if (device) {
            source->copyToHostTensor(result);
        } else {
            Backend::Info info;
            info.type    = MNN_FORWARD_CPU;
            auto backend = MNNGetExtraBackendCreator(MNN_FORWARD_CPU)->onCreate(info);
            backend->onCopyBuffer(source, result);
            delete backend;
        }
        return result;
    } else {
        return source->createHostTensorFromDevice(source, true);
    }
}

template <typename T>
static void copyTensorToFloat(const Tensor* source, double* dest) {
    auto srcData = source->host<T>();
    auto size    = source->elementSize();
    for (int i = 0; i < size; ++i) {
        dest[i] = srcData[i];
    }
}

static bool equals(const double* pa, const double* pb, size_t size, double tolerance, double epsilon, bool overall,
                   bool prints) {
    // get max if using overall torelance
    double max = fabs(pb[0]);
    if (overall) {
        for (int i = 1; i < size; i++) {
            max = std::max(max, fabs(pb[i]));
        }
    }

    // compare
    for (int i = 0; i < size; i++) {
        float va = pa[i], vb = pb[i];
        if (std::isinf(va) && std::isinf(vb)) {
            continue;
        }
        if (va < epsilon && vb < epsilon) {
            continue;
        }
        float div = overall ? max : fabsf(vb);
        if (fabsf(va - vb) / div > tolerance) {
            if (prints) {
                MNN_PRINT("%d: %f != %f\n", i, va, vb);
            }
            return false;
        }
    }
    return true;
}

bool TensorUtils::compareTensors(const Tensor* compare, const Tensor* expect, float tolerance, bool overall,
                                 bool printsErrors, bool printsTensors) {
    // type
    if (compare->getType().code != expect->getType().code || compare->getType().bits != expect->getType().bits) {
        if (printsErrors) {
            MNN_PRINT("NOT equal in type: %d/%d - %d/%d.\n", compare->getType().code, compare->getType().bits,
                      expect->getType().code, expect->getType().bits);
        }
        return false;
    }

    // dimensions
    if (compare->dimensions() != expect->dimensions()) {
        if (printsErrors) {
            MNN_PRINT("NOT equal in dimensions: %d - %d.\n", compare->dimensions(), expect->dimensions());
        }
        return false;
    }
    for (int i = 0; i < compare->dimensions(); i++) {
        if (compare->length(i) == expect->length(i)) {
            continue;
        }
        if (printsErrors) {
            MNN_PRINT("NOT equal in dimensions[%d]: %d - %d.\n", i, compare->length(i), expect->length(i));
        }
        return false;
    }

    // convert to host if needed
    auto a = createHostPlanar(compare), b = createHostPlanar(expect);

    // get value as double
    auto size = expect->elementSize();
    std::vector<double> expectValue(expect->elementSize(), 0.0f);
    std::vector<double> compareValue(compare->elementSize(), 0.0f);

    auto result = false;
    if (b->buffer().type.code == halide_type_uint) {
        switch (b->buffer().type.bits) {
            case 8:
                copyTensorToFloat<uint8_t>(a, compareValue.data());
                copyTensorToFloat<uint8_t>(b, expectValue.data());
                break;
            case 16:
                copyTensorToFloat<uint16_t>(a, compareValue.data());
                copyTensorToFloat<uint16_t>(b, expectValue.data());
                break;
            case 32:
                copyTensorToFloat<uint32_t>(a, compareValue.data());
                copyTensorToFloat<uint32_t>(b, expectValue.data());
                break;
            case 64:
                copyTensorToFloat<uint64_t>(a, compareValue.data());
                copyTensorToFloat<uint64_t>(b, expectValue.data());
                break;
            default:
                break;
        }
    } else if (b->buffer().type.code == halide_type_int) {
        switch (b->buffer().type.bits) {
            case 8:
                copyTensorToFloat<int8_t>(a, compareValue.data());
                copyTensorToFloat<int8_t>(b, expectValue.data());
                break;
            case 16:
                copyTensorToFloat<int16_t>(a, compareValue.data());
                copyTensorToFloat<int16_t>(b, expectValue.data());
                break;
            case 32:
                copyTensorToFloat<int32_t>(a, compareValue.data());
                copyTensorToFloat<int32_t>(b, expectValue.data());
                break;
            case 64:
                copyTensorToFloat<int64_t>(a, compareValue.data());
                copyTensorToFloat<int64_t>(b, expectValue.data());
                break;
            default:
                break;
        }
    } else if (b->buffer().type.code == halide_type_float) {
        switch (b->buffer().type.bits) {
#ifdef __FLT16_EPSILON__
            case 16:
                copyTensorToFloat<__fp16>(a, compareValue.data());
                copyTensorToFloat<__fp16>(b, expectValue.data());
                break;
#endif
            case 32:
                copyTensorToFloat<float>(a, compareValue.data());
                copyTensorToFloat<float>(b, expectValue.data());
                break;
            default:
                break;
        }
    } else {
        if (printsErrors) {
            MNN_PRINT("unsupported data type.");
        }
    }
    auto epsilon = FLT_EPSILON;
    if ((NULL != compareValue.data()) && (NULL != expectValue.data())) {
        result = equals(compareValue.data(), expectValue.data(), size, tolerance, epsilon, overall, printsErrors);
    }
    if (!result && printsTensors) {
        a->print();
        b->print();
    }

    // clean up
    if (a != compare) {
        delete a;
    }
    if (b != expect) {
        delete b;
    }
    return result;
}
} // namespace MNN
