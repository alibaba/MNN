#include "scheduler.hpp"
#include <math.h>
#include "core/Macro.h"
namespace MNN {
namespace DIFFUSION {

FlowMatchEulerScheduler::FlowMatchEulerScheduler(int trainTimestepsNum, float shift, bool useDynamicShifting)
    : mTrainTimestepsNum(trainTimestepsNum), mShift(shift), mUseDynamicShifting(useDynamicShifting) {
}

std::vector<float> FlowMatchEulerScheduler::get_sigmas(int inferenceSteps) const {
    if (inferenceSteps < 1) {
        return {0.0f};
    }
    
    // Match Python diffusers FlowMatchEulerDiscreteScheduler:
    // timesteps = np.linspace(num_train_timesteps, 1, num_inference_steps)
    // sigmas = timesteps / num_train_timesteps
    // This gives [1.0, 0.9474, 0.8948, ..., 0.001] for 20 steps
    float timestepMax = static_cast<float>(mTrainTimestepsNum);  // 1000
    float timestepMin = 1.0f;

    std::vector<float> sigmas(inferenceSteps + 1);
    for (int i = 0; i < inferenceSteps; ++i) {
        float frac = (inferenceSteps == 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(inferenceSteps - 1);
        float timestep = timestepMax + (timestepMin - timestepMax) * frac;
        float sigma = timestep / timestepMax;
        if (!mUseDynamicShifting && mShift != 1.0f) {
            // Apply shift: sigma = shift * sigma / (1 + (shift - 1) * sigma)
            sigma = mShift * sigma / (1.0f + (mShift - 1.0f) * sigma);
        }
        sigmas[i] = sigma;
    }
    sigmas[inferenceSteps] = 0.0f;  // Final sigma is always 0
    return sigmas;
}

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
