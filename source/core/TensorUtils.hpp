//
//  TensorUtils.hpp
//  MNN
//
//  Created by MNN on 2019/01/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TensorUtils_hpp
#define TensorUtils_hpp

#include <MNN/Tensor.hpp>
#include "Tensor_generated.h"

namespace MNN {
class Backend;

/** extra tensor info container */
struct Tensor::InsideDescribe {
public:
    /** dimension format */
    MNN_DATA_FORMAT dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
    /** buffer dimensions pointer holder */
    halide_dimension_t* dimensionStorage = nullptr;
    /** handle type */
    HandleDataType handleType = HANDLE_NONE;
    /** function used to free handle */
    void (*handleFreeFunction)(void*) = nullptr;

    /** for HOST tensor only. host memory is owned by tensor or not */
    bool ownHost = false;
    /** for DEVICE tensor only. backend used to manage tensor's device memory. */
    Backend* backend = nullptr;
    /** for DEVICE tensor only. */
    int useCount = 0;
    enum Usage {
        NORMAL,
        INPUT,
        OUTPUT,
        CONST,
        /** Whether the tensor is a trainable parameter. Trainable parameter should be stored in a different area. */
        TRAINABLE,
    };
    Usage usage = NORMAL;
};
typedef Tensor::InsideDescribe::Usage TensorUsage;

/** tensor utils */
class MNN_PUBLIC TensorUtils {
public:
    /**
     * @brief get extra tensor info.
     * @param tensor    given tensor.
     * @return extra tensor info.
     */
    static Tensor::InsideDescribe* getDescribe(const Tensor* tensor);

    /**
     * @brief copy shape from source tensor to dest tensor.
     * @param source        shape prodiver tensor.
     * @param dest          shape consumer tensor.
     * @param copyFormat    copy data format or not.
     */
    static void copyShape(const Tensor* source, Tensor* dest, bool copyFormat = false);

    /**
     * auto update tensor's strides according to extents and reorder flags.
     * @param tensor    given tensor.
     */
    static void setLinearLayout(Tensor* tensor);

    /**
     * @brief call handle free function to clear handle of tensor.
     * @param tensor    given tensor.
     */
    static void clearHandleData(Tensor* tensor);

    /**
     * @brief compare tensor to expected with tolerance.
     * @param compareTensor comparing tensor.
     * @param toTensor      expected tensor.
     * @param tolerance     tolerable error, any error less than this value will be ignored.
     *                      for integer types, compare with `abs(v1 - v2) > tolerance`;
     *                      for float types, see `overallTolerance`.
     * @param overall       for float types only. compare with `abs(v1 - v2) / max(abs(allExpectValues))` if true,
     *                      `abs(v1 - v2) / abs(v2)` otherwise.
     * @param printsError   print error data or not.
     * @param printsTensors print tensor data or not when meets error.
     * @return equals within tolerance or not.
     */
    static bool compareTensors(const Tensor* compareTensor, const Tensor* toTensor, float tolerance = 0,
                               bool overall = false, bool printsError = true, bool printsTensors = false);
};
} // namespace MNN

#endif /* TensorDescribe_hpp */
