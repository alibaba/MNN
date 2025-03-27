#include "scheduler.hpp"
#include <math.h>
#include "core/Macro.h"
namespace MNN {
namespace DIFFUSION {

static std::vector<float> linspace(float start, float end, int num) {
    std::vector<float> result(num);
    float step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}
static std::vector<float> cumulative_product(const std::vector<float> input) {
    std::vector<float> result(input.size());
    result[0] = input[0];
    for (int i = 1; i < input.size(); ++i) {
        result[i] = result[i - 1] * input[i];
    }
    return result;
}

PNDMScheduler::PNDMScheduler() {
    if(mBetaSchedule == "scaled_linear") {
        auto betas = linspace(std::sqrt(mBetaStart), std::sqrt(mBetaEnd), mTrainTimestepsNum);
        for (auto& beta : betas) {
            beta = 1 - beta * beta;
        }
        mAlphasCumProd = cumulative_product(betas);
    } else {
        MNN_ERROR("Error: not supported diffusion scheduler method\n");
    }
}

}
} // diffusion
