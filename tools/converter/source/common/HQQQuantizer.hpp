#pragma once
#include <MNN/MNNDefine.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <string>
#include <limits>
#include <MNN/expr/ExprCreator.hpp>

namespace MNN {
namespace Quantization {

/**
 * HQQ (Half-Quadratic Quantization) 量化器 C++ 实现
 * 基于半二次量化算法的权重量化和编码
 */
class HQQQuantizer {
public:
    struct QuantizationConfig {
        int bits = 4;
        int group_size = 64;
        bool optimize = true;
        float lp_norm = 0.7f;
        float beta = 10.0f;
        float kappa = 1.01;
        int iters = 20;
    };

    struct QuantizationResult {
        MNN::Express::VARP QW;
        MNN::Express::VARP SZ;
        QuantizationConfig config;               // 配置信息
        size_t elementSize = 0;
    };

private:
    QuantizationConfig mConfig;

public:
    explicit HQQQuantizer(const QuantizationConfig& config);
    
    /**
     * 量化权重矩阵
     * @param weights 输入权重数据
     * @param shape 权重形状 [height, width]
     * @return 量化结果
     */
    QuantizationResult quantize(const std::vector<float>& weights);
    
    /**
     * 反量化权重矩阵
     * @param result 量化结果
     * @return 反量化后的权重
     */
    MNN::Express::VARP dequantize(const QuantizationResult& result);

private:
    void optimize(MNN::Express::VARP& scale, MNN::Express::VARP& zero, MNN::Express::VARP WF);
};

} // namespace Quantization
} // namespace AliNN
