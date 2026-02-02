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
    SANA_DIFFUSION = 2,
    DIFFUSION_MODEL_USER
} DiffusionModelType;

class MNN_PUBLIC Diffusion {
public:
    Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode);
    virtual ~Diffusion();
    static Diffusion* createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode);
    virtual bool run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed, std::function<void(int)> progressCallback) = 0;

    // 统一的生成接口，支持所有Diffusion模型
    // 参数说明：
    //   input_embeds: 文本特征向量（来自LLM或Text Encoder）
    //                 - Sana: 来自Qwen3-0.6B LLM
    //                 - Stable Diffusion: 来自CLIP Text Encoder
    //   mode: 生成模式
    //         - "text2img": 文生图模式（从随机噪声开始）
    //         - "img2img": 图像编辑模式（从输入图像的latent开始）
    //   inputImagePath: 输入图像路径（img2img模式必需，text2img模式忽略）
    //   outputImagePath: 输出图像路径
    //   width: 输出图像宽度（必须是模型支持的尺寸，如512, 1024等）
    //   height: 输出图像高度（必须是模型支持的尺寸，如512, 1024等）
    //   iterNum: 推理步数（通过蒸馏可以用较少步数获得高质量结果）
    //   randomSeed: 随机种子（-1表示随机）
    //   use_cfg: 是否使用Classifier-Free Guidance
    //            需要input_embeds包含正负样本（batch_size=2）
    //   cfg_scale: CFG引导强度（仅use_cfg=true时生效，典型值：4.5-7.5）
    //   progressCallback: 进度回调函数
    virtual bool run(const VARP input_embeds, 
                    const std::string& mode,
                    const std::string& inputImagePath,
                    const std::string& outputImagePath,
                    int width,
                    int height,
                    int iterNum,
                    int randomSeed,
                    bool use_cfg,
                    float cfg_scale,
                    std::function<void(int)> progressCallback) = 0;
    
    virtual bool load() = 0;

protected:
    std::shared_ptr<Executor::RuntimeManager> runtime_manager_;
    std::vector<std::shared_ptr<Module>> mModules;
    
    std::string mModelPath;
    DiffusionModelType mModelType;
    /* 0 -> memory saving mode, for memory stictly limited application
        1 -> memory enough mode, for better image generation speed
        2 -> balance mode for memory and generation speed.
     */
    int mMemoryMode;
    MNNForwardType mBackendType;
};

}
}
#endif
