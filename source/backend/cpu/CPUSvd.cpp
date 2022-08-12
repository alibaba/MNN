//
//  CPUSvd.cpp
//  MNN
//
//  Created by MNN on 2022/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUSvd.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include <cmath>

namespace MNN {
static void orthogonal(float* at, float* vt, int i, int j, int row, int col, bool& pass) {
    auto ai = at + i * row;
    auto aj = at + j * row;
    auto vi = vt + i * col;
    auto vj = vt + j * col;
    float norm = 0.f, normi = 0.f, normj = 0.f;
    for (int i = 0; i < col; i++) {
        norm += ai[i] * aj[i];
        normi += ai[i] * ai[i];
        normj += aj[i] * aj[i];
    }
    constexpr float eps = std::numeric_limits<float>::epsilon() * 2;
    if (std::abs(norm) < eps * std::sqrt(normi * normj)) {
        return;
    }
    pass = false;
    float tao = (normi - normj) / (2.0 * norm);
    float tan = (tao < 0 ? -1 : 1) / (fabs(tao) + sqrt(1 + pow(tao, 2)));
    float cos = 1 / sqrt(1 + pow(tan, 2));
    float sin = cos * tan;
    bool swap = normi < normj;
    for (int i = 0; i < col; i++) {
        float nai = ai[i];
        float naj = aj[i];
        float nvi = vi[i];
        float nvj = vj[i];
        if (swap) {
            std::swap(nai, naj);
            std::swap(nvi, nvj);
        }
        ai[i] = nai * cos + naj * sin;
        aj[i] = naj * cos - nai * sin;
        vi[i] = nvi * cos + nvj * sin;
        vj[i] = nvj * cos - nvi * sin;
    }
}
ErrorCode CPUSvd::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mRow = inputs[0]->length(0);
    mCol = inputs[0]->length(1);
    mAt.reset(Tensor::createDevice<float>(std::vector<int>{mCol, mRow}));
    bool success = static_cast<CPUBackend*>(backend())->onAcquireBuffer(mAt.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Error for alloc memory for Alloc At\n");
        return OUT_OF_MEMORY;;
    }
    return NO_ERROR;
}

ErrorCode CPUSvd::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto A = inputs[0];
    auto W = outputs[0];
    auto U = outputs[1];
    auto Vt = outputs[2];    
    auto a = A->host<float>();
    auto w = W->host<float>();
    auto u = U->host<float>();
    auto at = mAt->host<float>();
    auto vt = Vt->host<float>();
    auto core = static_cast<CPUBackend*>(backend())->functions();
    // init at
    for (int i = 0; i < mCol; i++) {
        for (int j = 0; j < mRow; j++) {
            at[i * mRow + j] = a[j * mCol + i];
        }
    }
    // init vt
    for (int i = 0; i < mCol; i++) {
        for (int j = 0; j < mCol; j++) {
            vt[i * mCol + j] = (i == j);
        }
    }
    constexpr int max_iteration = 30;
    for (int iter = 0; iter < max_iteration; iter++) {
        bool pass = true;
        for (int i = 0; i < mCol; i++) {
            for (int j = i + 1; j < mCol; j++) {
                orthogonal(at, vt, i, j, mRow, mCol, pass);
            }
        }
        if (pass) break;
    }
    for (int i = 0; i < mCol; i++) {
        float norm = 0.f;
        for (int j = 0; j < mCol; j++) {
            auto tmp = at[i * mCol + j];
            norm += tmp * tmp;
        }
        norm = sqrt(norm);
        w[i] = norm;
    }
    for (int i = 0; i < mRow; i++) {
        for (int j = 0; j < mCol; j++) {
            u[i * mCol + j] = at[j * mCol + i] / w[j];
        }
    }
    return NO_ERROR;
}

class CPUSvdCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUSvd(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSvdCreator, OpType_Svd);
} // namespace MNN
