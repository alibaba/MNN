//
//  CPUTensorConvert.cpp
//  MNN
//
//  Created by MNN on 2018/08/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUTensorConvert.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"

namespace MNN {

template<typename T>
void NCHW2NHWC(const T* source, T* dest, int b, int c, int area) {
    int sourceBatchsize = c * area;
    int destBatchSize   = sourceBatchsize;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        for (int i = 0; i < area; ++i) {
            auto srcArea = srcBatch + i;
            auto dstArea = dstBatch + i * c;
            for (int ci = 0; ci < c; ++ci) {
                dstArea[ci] = srcArea[ci * area];
            }
        }
    }
}

template<typename T>
void NHWC2NCHW(const T* source, T* dest, int b, int c, int area) {
    int sourceBatchsize = c * area;
    int destBatchSize   = sourceBatchsize;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        for (int i = 0; i < area; ++i) {
            auto srcArea = srcBatch + i * c;
            auto dstArea = dstBatch + i;
            for (int ci = 0; ci < c; ++ci) {
                dstArea[ci * area] = srcArea[ci];
            }
        }
    }
}
typedef void(*PackProc)(void* dst, const void* src, size_t area, size_t depth, int* areaOffset);

ErrorCode CPUTensorConverter::convert(const void* inputRaw, void* outputRaw, MNN_DATA_FORMAT source, MNN_DATA_FORMAT dest, int batch, int area, int channel, int bitLength, const CoreFunctions* core, int tId, int numberThread) {
    // the case when source and dest data layout are the same
    // This case occurs in BackendTest of BF16 data.
    if(source == dest) {
        if (tId == 0) {
            ::memcpy(outputRaw, inputRaw, batch * area * channel * bitLength);
        }
        return NO_ERROR;
    }
    if (MNN_DATA_FORMAT_NHWC == source && MNN_DATA_FORMAT_NCHW == dest) {
        if (tId == 0) {
            switch (bitLength) {
                case 1:
                    NHWC2NCHW((int8_t*)inputRaw, (int8_t*)outputRaw, batch, channel, area);
                    break;
                case 2:
                    NHWC2NCHW((int16_t*)inputRaw, (int16_t*)outputRaw, batch, channel, area);
                    break;
                case 4:
                    NHWC2NCHW((float*)inputRaw, (float*)outputRaw, batch, channel, area);
                    break;
                default:
                    break;
            }
        }
        return NO_ERROR;
    }
    if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NHWC == dest) {
        if (tId == 0) {
            switch (bitLength) {
                case 1:
                    NCHW2NHWC((int8_t*)inputRaw, (int8_t*)outputRaw, batch, channel, area);
                    break;
                case 2:
                    NCHW2NHWC((int16_t*)inputRaw, (int16_t*)outputRaw, batch, channel, area);
                    break;
                case 4:
                    NCHW2NHWC((float*)inputRaw, (float*)outputRaw, batch, channel, area);
                    break;
                default:
                    break;
            }
        }
        return NO_ERROR;
    }
    // Need Pack
    PackProc proc = nullptr;
    int inside = area;
    int outside = batch;
    if (MNN_DATA_FORMAT_NHWC == source || MNN_DATA_FORMAT_NHWC == dest) {
        inside = 1;
        outside = batch * area;
    }
    //MNN_PRINT("bytes = %d, from %d -> %d, %d - %d - %d\n", bitLength, source, dest, inside, outside, channel);
    if (MNN_DATA_FORMAT_NC4HW4 == source) {
        if (1 == inside) {
            int offset[2] = {
                outside,
                outside
            };
            int step = UP_DIV(outside, numberThread);
            int start = tId * step;
            int end = std::min(start + step, outside);
            if (end <= start) {
                return NO_ERROR;
            }
            auto inputStart = (int8_t*)inputRaw + (start * core->pack * bitLength);
            auto outputStart = (int8_t*)outputRaw + (start * channel * bitLength);
            if (core->bytes == bitLength) {
                proc = decltype(proc)(core->MNNUnpackCUnitTranspose);
            } else if (bitLength == 1) {
                proc = decltype(proc)(core->MNNUnpackCUnitTransposeInt8);
            } else if (bitLength == 2) {
                proc = decltype(proc)(core->MNNUnpackCUnitTransposeInt16);
            }
            if (nullptr == proc) {
                return NOT_SUPPORT;
            }
            proc((float*)outputStart, (const float*)inputStart, end - start, channel, offset);
        } else {
            if (core->bytes == bitLength) {
                proc = decltype(proc)(core->MNNUnpackCUnit);
            } else if (bitLength == 1) {
                proc = decltype(proc)(core->MNNUnpackCUnitInt8);
            } else if (bitLength == 2) {
                proc = decltype(proc)(core->MNNUnpackCUnitInt16);
            }
            if (nullptr == proc) {
                return NOT_SUPPORT;
            }
            if (batch != 1) {
                // Divide in batch
                int offset[2] = {
                    outside * inside,
                    area
                };
                int step = UP_DIV(batch, numberThread);
                int start = tId * step;
                int end = std::min(start + step, batch);
                if (end <= start) {
                    return NO_ERROR;
                }
                for (int v=start; v<end; ++v) {
                    auto inputStart = (int8_t*)inputRaw + (v * core->pack * bitLength * area);
                    auto outputStart = (int8_t*)outputRaw + (v * channel * bitLength * area);
                    proc((float*)outputStart, (const float*)inputStart, area, channel, offset);
                }
            } else {
                // Divide in area
                int offset[2] = {
                    area,
                    area
                };
                int step = UP_DIV(area, numberThread);
                int start = tId * step;
                int end = std::min(start + step, area);
                if (end <= start) {
                    return NO_ERROR;
                }
                auto inputStart = (int8_t*)inputRaw + (start * core->pack * bitLength);
                auto outputStart = (int8_t*)outputRaw + (start * bitLength);
                proc((float*)outputStart, (const float*)inputStart, end - start, channel, offset);
            }
        }
        return NO_ERROR;
    }
    if (MNN_DATA_FORMAT_NC4HW4 == dest) {
        if (1 == inside) {
            int offset[2] = {
                outside,
                outside
            };
            int step = UP_DIV(outside, numberThread);
            int start = tId * step;
            int end = std::min(start + step, outside);
            if (end <= start) {
                return NO_ERROR;
            }
            if (core->bytes == bitLength) {
                proc = decltype(proc)(core->MNNPackCUnitTranspose);
            } else if (bitLength == 1) {
                proc = decltype(proc)(core->MNNPackCUnitTransposeInt8);
            } else if (bitLength == 2) {
                proc = decltype(proc)(core->MNNPackCUnitTransposeInt16);
            }
            if (nullptr == proc) {
                return NOT_SUPPORT;
            }
            auto outputStart = (int8_t*)outputRaw + (start * core->pack * bitLength);
            auto inputStart = (int8_t*)inputRaw + (start * channel * bitLength);
            proc(outputStart, inputStart, end - start, channel, offset);
        } else {
            if (core->bytes == bitLength) {
                proc = decltype(proc)(core->MNNPackCUnit);
            } else if (bitLength == 1) {
                proc = decltype(proc)(core->MNNPackCUnitInt8);
            } else if (bitLength == 2) {
                proc = decltype(proc)(core->MNNPackCUnitInt16);
            }
            if (nullptr == proc) {
                return NOT_SUPPORT;
            }
            if (batch != 1) {
                // Divide in batch
                int offset[2] = {
                    area,
                    outside * inside
                };
                int step = UP_DIV(batch, numberThread);
                int start = tId * step;
                int end = std::min(start + step, batch);
                if (end <= start) {
                    return NO_ERROR;
                }
                for (int v=start; v<end; ++v) {
                    auto outputStart = (int8_t*)outputRaw + (v * core->pack * bitLength * area);
                    auto inputStart = (int8_t*)inputRaw + (v * channel * bitLength * area);
                    proc((float*)outputStart, (const float*)inputStart, area, channel, offset);
                }
            } else {
                // Divide in area
                int offset[2] = {
                    area,
                    area
                };
                int step = UP_DIV(area, numberThread);
                int start = tId * step;
                int end = std::min(start + step, area);
                if (end <= start) {
                    return NO_ERROR;
                }
                auto outputStart = (int8_t*)outputRaw + (start * core->pack * bitLength);
                auto inputStart = (int8_t*)inputRaw + (start * bitLength);
                proc((float*)outputStart, (const float*)inputStart, end - start, channel, offset);
            }
        }
        return NO_ERROR;
    }
    return NO_ERROR;
}

std::tuple<int, int, int> CPUTensorConverter::splitDimensions(const halide_buffer_t& ib, MNN_DATA_FORMAT source) {
    int area = 1, batch = ib.dim[0].extent, channel;
    if (source == MNN_DATA_FORMAT_NC4HW4 || source == MNN_DATA_FORMAT_NCHW) {
        channel = ib.dim[1].extent;
        for (int axis = 2; axis < ib.dimensions; ++axis) {
            area *= ib.dim[axis].extent;
        }
    } else {
        channel = ib.dim[ib.dimensions - 1].extent;
        for (int axis = 1; axis < ib.dimensions - 1; ++axis) {
            area *= ib.dim[axis].extent;
        }
    }
    return std::make_tuple(batch, area, channel);
}

static int _getBytes(const CoreFunctions* core, const Tensor* output) {
    auto bytes = output->getType().bytes();
    auto quant = TensorUtils::getDescribe(output)->quantAttr.get();
    if (output->getType().code == halide_type_float) {
        bytes = core->bytes;
    }
    if (nullptr != quant && TensorUtils::getDescribe(output)->type == DataType_DT_INT8) {
        bytes = 1;
    }
    return bytes;
}
ErrorCode CPUTensorConverter::convert(const Tensor* input, const Tensor* output, const CoreFunctions* core, int tId, int numberThread) {
    auto ib     = input->buffer();
    auto ob     = output->buffer();
    auto source = TensorUtils::getDescribe(input)->dimensionFormat;
    auto dest   = TensorUtils::getDescribe(output)->dimensionFormat;
    if (nullptr == core) {
        core = MNNGetCoreFunctions();
    }
    size_t bitLength = _getBytes(core, input);
    if (ib.dimensions <= 1 || source == dest) {
        size_t dataSize = 1;
        for (int i = 0; i < input->dimensions(); i++) {
            int currentDimSize = input->length(i);
            if (source == MNN_DATA_FORMAT_NC4HW4 && 1 == i) {
                currentDimSize = UP_DIV(currentDimSize, core->pack) * core->pack;
            }
            dataSize *= currentDimSize;
        }
        // printf("convert # dataSize, bitLength = %d, %d\n", dataSize, bitLength);
        // fflush(stdout);
        ::memcpy(ob.host, ib.host, dataSize * bitLength);
        return NO_ERROR;
    }
    if (source == MNN_DATA_FORMAT_UNKNOWN || dest == MNN_DATA_FORMAT_UNKNOWN) {
        MNN_ERROR("unknown data format!\nsrc: %s, dst: %s\n", EnumNameMNN_DATA_FORMAT(source), EnumNameMNN_DATA_FORMAT(dest));
        return INVALID_VALUE;
    }
    auto tup = splitDimensions(ib, source);
    int area = std::get<1>(tup), batch = std::get<0>(tup), channel = std::get<2>(tup);
    auto code = convert(ib.host, ob.host, source, dest, batch, area, channel, bitLength, core, tId, numberThread);
    if (NO_ERROR != code) {
        MNN_ERROR("Error in CPUTensorConver\n");
        return code;
    }
    return NO_ERROR;
}

} // namespace MNN
