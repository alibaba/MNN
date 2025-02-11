//
//  diffusion.hpp
//
//  Created by MNN on 2025/01/12.
//  MNN
//
#ifndef MNN_DIFFUSION_HPP
#define MNN_DIFFUSION_HPP

#include <map>
#include <vector>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Module.hpp>

using namespace MNN;
using namespace MNN::Express;

namespace MNN {
namespace DIFFUSION {

class Tokenizer;
typedef enum {
    STABLE_DIFFUSION_1_5 = 0,
    STABLE_DIFFUSION_TAIYI_CHINESE = 1,
    DIFFUSION_MODEL_USER
} DiffusionModelType;

class MNN_PUBLIC Diffusion {
public:
    Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int iterationNum);
    virtual ~Diffusion();
    static Diffusion* createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int iterationNum);

    bool run(const std::string prompt, const std::string imagePath, std::function<void(int)> progressCallback);
    bool load();
private:
    VARP step_plms(VARP sample, VARP model_output, int index);
    VARP text_encoder(const std::vector<int>& ids);
    VARP unet(VARP text_embeddings, std::function<void(int)> progressCallback);
    VARP vae_decoder(VARP latent);
private:
    std::shared_ptr<Executor::RuntimeManager> runtime_manager_;
    std::vector<std::shared_ptr<Module>> mModules;
    // step_plms
    std::vector<int> mTimeSteps;
    std::vector<float> mAlphas;
    std::vector<VARP> mEts;
    VARP mSample;
    VARP mLatentVar, mPromptVar, mTimestepVar, mSampleVar;
    std::vector<float> mInitNoise;
    
private:
    std::string mModelPath;
    DiffusionModelType mModelType;
    int mMaxTextLen = 77;
    int mMemoryMode;
    int mIterationNum;
    MNNForwardType mBackendType;
    std::unique_ptr<Tokenizer> mTokenizer;
};

}
}
#endif
