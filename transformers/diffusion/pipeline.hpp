#include <map>
#include <vector>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Module.hpp>

using namespace MNN;
using namespace MNN::Express;

namespace diffusion {

typedef enum {
    STABLE_DIFFUSION_1_5 = 0,
    STABLE_DIFFUSION_TAIYI_CHINESE = 1,
    DIFFUSION_MODEL_USER
} DiffusionModelType;
    
class Pipeline {
public:
    Pipeline(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode);
    ~Pipeline() = default;
    bool run(const std::string& prompt, const std::string& imagePath);
private:
    bool load_modules();
    VARP step_plms(VARP sample, VARP model_output, int index);
    VARP text_encoder(const std::vector<int>& ids);
    VARP unet(VARP text_embeddings);
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

private:
    std::string mModelPath;
    DiffusionModelType mModelType;
    int mMaxTextLen = 77;
    int mMemoryMode;
    MNNForwardType mBackendType;
};

}
