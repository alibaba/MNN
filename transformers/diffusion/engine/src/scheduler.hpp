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
