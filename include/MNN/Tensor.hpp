//
//  Tensor.hpp
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Tensor_hpp
#define Tensor_hpp

#include <vector>
#include <MNN/HalideRuntime.h>
#include <MNN/MNNDefine.h>
#define MNN_MAX_TENSOR_DIM 6

namespace MNN {

/**
 * data container.
 * data for host tensor is saved in `host` field. its memory is allocated malloc directly.
 * data for device tensor is saved in `deviceId` field. its memory is allocated by session's backend.
 * usually, device tensors are created by engine (like net, session).
 * meanwhile, host tensors could be created by engine or user.
 */
class MNN_PUBLIC Tensor {
public:
    struct InsideDescribe;

    /** dimension type used to create tensor */
    enum DimensionType {
        /** for tensorflow net type. uses NHWC as data format. */
        TENSORFLOW,
        /** for caffe net type. uses NCHW as data format. */
        CAFFE,
        /** for caffe net type. uses NC4HW4 as data format. */
        CAFFE_C4
    };

    /** handle type */
    enum HandleDataType {
        /** default handle type */
        HANDLE_NONE = 0,
        /** string handle type */
        HANDLE_STRING = 1
    };

public:
    /**
     * @brief create a tensor with dimension size and type without acquire memory for data.
     * @param dimSize   dimension size.
     * @param type      dimension type.
     */
    Tensor(int dimSize = 4, DimensionType type = CAFFE);

    /**
     * @brief create a tensor with same shape as given tensor.
     * @param tensor        shape provider.
     * @param type          dimension type.
     * @param allocMemory   acquire memory for data or not.
     * @warning tensor data won't be copied.
     */
    Tensor(const Tensor* tensor, DimensionType type = CAFFE, bool allocMemory = true);

    /** deinitializer */
    ~Tensor();

private:
    // remove all assignment operator
    Tensor(const Tensor& tensor)  = delete;
    Tensor(const Tensor&& tensor) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor& operator=(const Tensor&&) = delete;

public:
    /**
     * @brief create tensor with shape, data type and dimension type.
     * @param shape     tensor shape.
     * @param type      data type.
     * @param dimType   dimension type.
     * @return created tensor.
     * @warning memory for data won't be acquired. call backend's onAcquireBuffer to get memory ready.
     */
    static Tensor* createDevice(const std::vector<int>& shape, halide_type_t type, DimensionType dimType = TENSORFLOW);

    /**
     * @brief create tensor with shape and dimension type. data type is represented by `T`.
     * @param shape     tensor shape.
     * @param dimType   dimension type.
     * @return created tensor.
     * @warning memory for data won't be acquired. call backend's onAcquireBuffer to get memory ready.
     */
    template <typename T>
    static Tensor* createDevice(const std::vector<int>& shape, DimensionType dimType = TENSORFLOW) {
        return createDevice(shape, halide_type_of<T>(), dimType);
    }

    /**
     * @brief create tensor with shape, data type, data and dimension type.
     * @param shape     tensor shape.
     * @param type      data type.
     * @param data      data to save.
     * @param dimType   dimension type.
     * @return created tensor.
     */
    static Tensor* create(const std::vector<int>& shape, halide_type_t type, void* data = NULL,
                          DimensionType dimType = TENSORFLOW);

    /**
     * @brief create tensor with shape, data and dimension type. data type is represented by `T`.
     * @param shape     tensor shape.
     * @param data      data to save.
     * @param dimType   dimension type.
     * @return created tensor.
     */
    template <typename T>
    static Tensor* create(const std::vector<int>& shape, void* data = NULL, DimensionType dimType = TENSORFLOW) {
        return create(shape, halide_type_of<T>(), data, dimType);
    }

public:
    /**
     * @brief for DEVICE tensor, copy data from given host tensor.
     * @param hostTensor    host tensor, the data provider.
     * @return true for DEVICE tensor, and false for HOST tensor.
     */
    bool copyFromHostTensor(const Tensor* hostTensor);

    /**
     * @brief for DEVICE tensor, copy data to given host tensor.
     * @param hostTensor    host tensor, the data consumer.
     * @return true for DEVICE tensor, and false for HOST tensor.
     */
    bool copyToHostTensor(Tensor* hostTensor) const;

    /**
     * @brief create HOST tensor from DEVICE tensor, with or without data copying.
     * @param deviceTensor  given device tensor.
     * @param copyData      copy data or not.
     * @return created host tensor.
     */
    static Tensor* createHostTensorFromDevice(const Tensor* deviceTensor, bool copyData = true);

public:
    const halide_buffer_t& buffer() const {
        return mBuffer;
    }
    halide_buffer_t& buffer() {
        return mBuffer;
    }

    /**
     * @brief get dimension type.
     * @return dimension type.
     */
    DimensionType getDimensionType() const;

    /**
     * @brief handle data type. used when data type code is halide_type_handle.
     * @return handle data type.
     */
    HandleDataType getHandleDataType() const;

    /**
     * @brief set data type.
     * @param type data type defined in 'Type_generated.h'.
     */
    void setType(int type);

    /**
     * @brief get data type.
     * @return data type.
     */
    inline halide_type_t getType() const {
        return mBuffer.type;
    }

    /**
     * @brief visit host memory, data type is represented by `T`.
     * @return data point in `T` type.
     */
    template <typename T>
    T* host() const {
        return (T*)mBuffer.host;
    }

    /**
     * @brief visit device memory.
     * @return device data ID. what the ID means varies between backends.
     */
    uint64_t deviceId() const {
        return mBuffer.device;
    }

public:
    int dimensions() const {
        return mBuffer.dimensions;
    }

    /**
     * @brief get all dimensions' extent.
     * @return dimensions' extent.
     */
    std::vector<int> shape() const;

    /**
     * @brief calculate number of bytes needed to store data taking reordering flag into account.
     * @return bytes needed to store data
     */
    int size() const;

    /**
     * @brief calculate number of elements needed to store data taking reordering flag into account.
     * @return elements needed to store data
     */
    inline int elementSize() const {
        return size() / mBuffer.type.bytes();
    }

public:
    inline int width() const {
        if (getDimensionType() == TENSORFLOW) {
            return mBuffer.dim[2].extent;
        }

        return mBuffer.dim[3].extent;
    }
    inline int height() const {
        if (getDimensionType() == TENSORFLOW) {
            return mBuffer.dim[1].extent;
        }
        return mBuffer.dim[2].extent;
    }
    inline int channel() const {
        if (getDimensionType() == TENSORFLOW) {
            return mBuffer.dim[3].extent;
        }
        return mBuffer.dim[1].extent;
    }
    inline int batch() const {
        return mBuffer.dim[0].extent;
    }

    // visit dimension's extent & stride
    inline int stride(int index) const {
        return mBuffer.dim[index].stride;
    }
    inline int length(int index) const {
        return mBuffer.dim[index].extent;
    }
    inline void setStride(int index, int stride) {
        mBuffer.dim[index].stride = stride;
    }
    inline void setLength(int index, int length) {
        mBuffer.dim[index].extent = length;
    }

public:
    /**
     * @brief print tensor data. for DEBUG use only.
     */
    void print() const;
    
    /**
     *@brief print tensor shape
     */
    void printShape() const;

private:
    halide_buffer_t mBuffer;
    struct InsideDescribe* mDescribe;

private:
    friend class TensorUtils;
};
} // namespace MNN

#endif /* Tensor_hpp */
