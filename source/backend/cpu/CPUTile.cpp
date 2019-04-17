//
//  CPUTile.cpp
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUTile.hpp"
#include "CPUBackend.hpp"

namespace MNN {

template <typename T>
static void CopyMultipleTimes(const T* inData, int32_t inSize, int32_t multiplier, T* outData) {
    for (int i = 0; i < multiplier; ++i) {
        const T* inEnd = inData + inSize;
        T* newOutData  = std::copy(inData, inEnd, outData);
        inData         = outData;
        outData        = newOutData;
    }
}

template <typename T, typename M>
std::pair<int, int> TileOneDimension(const Tensor* inTensor, const T* inData, const M* multipliers, T* outData,
                                     int dimension) {
    const int dimensionSize = inTensor->buffer().dim[dimension].extent;
    if (dimension == inTensor->buffer().dimensions - 1) {
        CopyMultipleTimes(inData, dimensionSize, multipliers[dimension], outData);
        return std::make_pair(dimensionSize, dimensionSize * static_cast<int>(multipliers[dimension]));
    }

    int totalStrideSize      = 0;
    int totalTiledStrideSize = 0;
    const T* copyFromData    = inData;
    T* copyToData            = outData;
    for (int i = 0; i < dimensionSize; i++) {
        int strideSize      = 0;
        int tiledStrideSize = 0;
        std::tie(strideSize, tiledStrideSize) =
            TileOneDimension(inTensor, copyFromData, multipliers, copyToData, dimension + 1);
        copyFromData += strideSize;
        copyToData += tiledStrideSize;
        totalStrideSize += strideSize;
        totalTiledStrideSize += tiledStrideSize;
    }
    CopyMultipleTimes(outData, totalTiledStrideSize, multipliers[dimension] - 1, outData + totalTiledStrideSize);
    return std::make_pair(totalStrideSize, totalTiledStrideSize * multipliers[dimension]);
}

CPUTile::CPUTile(Backend* b, const MNN::Op* op) : MNN::Execution(b) {
    // nothing to do
}

ErrorCode CPUTile::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input                 = inputs[0];
    const float* inData        = input->host<float>();
    const int32_t* multipliers = inputs[1]->host<int32_t>();
    float* outData             = outputs[0]->host<float>();
    TileOneDimension<float, int32_t>(input, inData, multipliers, outData, 0);

    return NO_ERROR;
}

class CPUTileCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUTile(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUTileCreator, OpType_Tile);

} // namespace MNN
