#ifndef MNN_QNN_TENSOR_CONVERT_HPP
#define MNN_QNN_TENSOR_CONVERT_HPP

#include <MNN/ErrorCode.hpp>
#include <MNN/Tensor.hpp>
#include "core/TensorUtils.hpp"
#include <cstring>
#include <limits>

namespace MNN {
// QNN host-boundary conversion; it does not depend on CPU/Arm82 private kernels.
struct QnnHostShape {
    size_t batch = 1, channel = 1, area = 1, bytes = 0;
    MNN_DATA_FORMAT format = MNN_DATA_FORMAT_NCHW;
};
inline bool qnnCheckedMultiply(size_t a, size_t b, size_t& result) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) return false;
    result = a * b;
    return true;
}
inline bool qnnHostShape(const Tensor* tensor, QnnHostShape& shape) {
    shape = QnnHostShape{};
    if (tensor == nullptr || tensor->dimensions() < 0 || tensor->dimensions() > 8) return false;
    shape.format = TensorUtils::getDescribe(tensor)->dimensionFormat;
    if (shape.format != MNN_DATA_FORMAT_NCHW && shape.format != MNN_DATA_FORMAT_NHWC &&
        shape.format != MNN_DATA_FORMAT_NC4HW4) return false;
    const int rank = tensor->dimensions();
    if (rank < 2) shape.format = MNN_DATA_FORMAT_NCHW;
    for (int i = 0; i < rank; ++i) if (tensor->length(i) <= 0) return false;
    if (rank > 0) shape.batch = tensor->length(0);
    if (rank > 1) {
        const int axis = shape.format == MNN_DATA_FORMAT_NHWC ? rank - 1 : 1;
        shape.channel = tensor->length(axis);
        for (int i = 1; i < rank; ++i) {
            if (i != axis && !qnnCheckedMultiply(shape.area, tensor->length(i), shape.area)) return false;
        }
    }
    size_t channel = shape.channel;
    if (shape.format == MNN_DATA_FORMAT_NC4HW4) channel = (channel + 3) / 4 * 4;
    size_t count = 0;
    return qnnCheckedMultiply(shape.batch, channel, count) &&
           qnnCheckedMultiply(count, shape.area, count) &&
           qnnCheckedMultiply(count, tensor->getType().bytes(), shape.bytes) && shape.bytes != 0;
}
inline size_t qnnHostOffset(const QnnHostShape& shape, size_t n, size_t c, size_t p) {
    if (shape.format == MNN_DATA_FORMAT_NHWC) return (n * shape.area + p) * shape.channel + c;
    if (shape.format == MNN_DATA_FORMAT_NC4HW4)
        return ((n * ((shape.channel + 3) / 4) + c / 4) * shape.area + p) * 4 + c % 4;
    return (n * shape.channel + c) * shape.area + p;
}
inline ErrorCode qnnConvertTensor(const Tensor* source, const Tensor* destination) {
    QnnHostShape a, b;
    if (!qnnHostShape(source, a) || !qnnHostShape(destination, b) ||
        source->getType() != destination->getType() || a.batch != b.batch ||
        a.channel != b.channel || a.area != b.area || source->host<uint8_t>() == nullptr ||
        destination->host<uint8_t>() == nullptr) return INPUT_DATA_ERROR;
    const auto* input = source->host<uint8_t>();
    auto* output = destination->host<uint8_t>();
    if (a.format == b.format) {
        std::memmove(output, input, a.bytes);
        return NO_ERROR;
    }
    if (input == output) return NOT_SUPPORT;
    if (b.format == MNN_DATA_FORMAT_NC4HW4) std::memset(output, 0, b.bytes);
    const size_t width = source->getType().bytes();
    for (size_t n = 0; n < a.batch; ++n)
        for (size_t c = 0; c < a.channel; ++c)
            for (size_t p = 0; p < a.area; ++p)
                std::memcpy(output + qnnHostOffset(b, n, c, p) * width,
                            input + qnnHostOffset(a, n, c, p) * width, width);
    return NO_ERROR;
}
} // namespace MNN
#endif
