//
//  diffusion_sd35.hpp
//
//  Created by zlaa on 2025/12/18.
//

#ifndef MNN_DIFFUSION_SD35_HPP
#define MNN_DIFFUSION_SD35_HPP

#include <map>
#include <vector>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Module.hpp>
#include "diffusion/diffusion.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace MNN {
namespace DIFFUSION {

class MNN_PUBLIC DiffusionSD35 {
public:
    DiffusionSD35(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode);
    virtual ~DiffusionSD35();
    static DiffusionSD35* createDiffusionSD35(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode);

    bool run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed, std::function<void(int)> progressCallback);
    bool load();

private:
    // Returns {prompt_embeds, pooled_prompt_embeds}
    std::pair<VARP, VARP> encode_prompt(const std::string& prompt);
    
    VARP transformer(VARP hidden_states, VARP encoder_hidden_states, VARP pooled_projections, VARP timestep);
    VARP vae_decoder(VARP latent);
    VARP step_flow_match(VARP sample, VARP model_output, int index, int num_steps);

private:
    std::shared_ptr<Executor::RuntimeManager> runtime_manager_;
    std::vector<std::shared_ptr<Module>> mModules; 
    // mModules[0]: text_encoder
    // mModules[1]: text_encoder_2
    // mModules[2]: text_encoder_3
    // mModules[3]: transformer
    // mModules[4]: vae_decoder

    std::vector<int> mTimeSteps;
    VARP mLatentVar;
    
private:
    std::string mModelPath;
    DiffusionModelType mModelType;
    int mMaxTextLen = 77;
    int mMaxTextLenT5 = 256; // T5 usually 256 or 512
    int mMemoryMode;
    MNNForwardType mBackendType;
    
    std::unique_ptr<Tokenizer> mTokenizer1;
    std::unique_ptr<Tokenizer> mTokenizer2;
    // std::unique_ptr<Tokenizer> mTokenizer3; // T5 not supported yet
};

}
}
#endif
