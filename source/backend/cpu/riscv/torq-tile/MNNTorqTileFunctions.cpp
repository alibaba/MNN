// Copyright © 2026, Alibaba Group Holding Limited

#include "MNNTorqTileFunctions.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"

namespace MNN {

void MNNTorqTileInitializeFunctions(CoreFunctions* core) {
    if (!core->supportRVV || core->bytes != 4 || core->pack != 4 || core->matmulBytes != 0) {
        return;
    }
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    if (eP != 16 || lP != 1 || hP != 4) {
        return;
    }
    core->MNNPackedMatMul = MNNPackedMatMulFP32_TorqTile;
    core->MNNPackedMatMulRemain = MNNPackedMatMulRemainFP32_TorqTile;
}

} // namespace MNN
