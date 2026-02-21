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

std::vector<float> FlowMatchEulerScheduler::get_sigmas_dynamic(int inferenceSteps, int imageSeqLen,
                                                                 int baseSeqLen, int maxSeqLen,
                                                                 float baseShift, float maxShift) const {
    if (inferenceSteps < 1) return {0.0f};

    // Step 1: calculate_shift (Python: pipeline_z_image.py calculate_shift)
    // mu = image_seq_len * m + b,  m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    float m = (maxShift - baseShift) / static_cast<float>(maxSeqLen - baseSeqLen);
    float b = baseShift - m * static_cast<float>(baseSeqLen);
    float mu = static_cast<float>(imageSeqLen) * m + b;
    MNN_PRINT("[ZImage Scheduler] image_seq_len=%d, mu=%.4f\n", imageSeqLen, mu);

    // Step 2: linspace from sigma_max to sigma_min=0 (Python: sigma_min=0.0 set in pipeline)
    // sigma_max = shift * 1.0 / (1 + (shift-1)*1.0) after static shift, but for dynamic we use
    // the raw linspace from 1.0 to 0.0 then apply time_shift_exponential
    std::vector<float> sigmas(inferenceSteps + 1);
    for (int i = 0; i < inferenceSteps; ++i) {
        // linspace from 1.0 to 0.0 (sigma_min=0), N points
        float t = (inferenceSteps == 1) ? 1.0f :
                  1.0f - static_cast<float>(i) / static_cast<float>(inferenceSteps - 1);
        // Step 3: exponential time_shift: exp(mu) / (exp(mu) + (1/t - 1)^1)
        // = exp(mu) * t / (exp(mu)*t + 1 - t)
        float exp_mu = expf(mu);
        float shifted;
        if (t <= 0.0f) {
            shifted = 0.0f;
        } else if (t >= 1.0f) {
            shifted = 1.0f;
        } else {
            shifted = exp_mu / (exp_mu + (1.0f / t - 1.0f));
        }
        sigmas[i] = shifted;
    }
    sigmas[inferenceSteps] = 0.0f;
    MNN_PRINT("[ZImage Scheduler] sigmas[0]=%.4f sigmas[1]=%.4f sigmas[-1]=%.4f\n",
              sigmas[0], inferenceSteps > 1 ? sigmas[1] : 0.0f, sigmas[inferenceSteps - 1]);
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
