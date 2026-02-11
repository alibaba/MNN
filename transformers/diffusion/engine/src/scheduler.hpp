#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

#ifndef MNN_DIFFUSION_SCHEDULER_HPP
#define MNN_DIFFUSION_SCHEDULER_HPP

namespace MNN {
namespace DIFFUSION {
    
class Scheduler {
public:
    Scheduler() = default;
    virtual ~Scheduler() = default;
    virtual std::vector<float> get_alphas() = 0;
};

class FlowMatchEulerScheduler {
public:
    FlowMatchEulerScheduler(int trainTimestepsNum = 1000, float shift = 3.0f, bool useDynamicShifting = false);
    std::vector<float> get_sigmas(int inferenceSteps) const;
    // Dynamic shifting: exponential time_shift with mu computed from image_seq_len
    // Matches Python ZImagePipeline: calculate_shift + time_shift_exponential + sigma_min=0
    std::vector<float> get_sigmas_dynamic(int inferenceSteps, int imageSeqLen,
                                          int baseSeqLen = 256, int maxSeqLen = 4096,
                                          float baseShift = 0.5f, float maxShift = 1.15f) const;
private:
    int mTrainTimestepsNum;
    float mShift;
    bool mUseDynamicShifting;
};

class PNDMScheduler : public Scheduler{
public:
    PNDMScheduler();
    std::vector<float> get_alphas() {
        return mAlphasCumProd;
    }
private:
    int mTrainTimestepsNum = 1000;
    float mBetaStart = 0.00085;
    float mBetaEnd = 0.012;
    int mStepsOffset = 1;
    std::string mBetaSchedule = "scaled_linear";
    std::vector<float> mAlphasCumProd;
};
}
} // diffusion
#endif
