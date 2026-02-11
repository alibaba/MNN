#include <random>
#include <fstream>
#include <chrono>
#include "diffusion/stable_diffusion.hpp"
#include "tokenizer.hpp"
#include "scheduler.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <cv/cv.hpp>
#include <fstream>
#include <sstream>
#include <MNN/expr/ExecutorScope.hpp>

#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#endif

// #define MNN_DUMP_DATA

using namespace CV;

namespace MNN {
namespace DIFFUSION {

StableDiffusion::StableDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode)
    : Diffusion(modelPath, modelType, backendType, memoryMode) {
    if(modelType == STABLE_DIFFUSION_1_5) {
        mMaxTextLen = 77;
    } else if(modelType == STABLE_DIFFUSION_TAIYI_CHINESE) {
        mMaxTextLen = 512;
    }
    // compute timesteps alphas
    std::unique_ptr<Scheduler> scheduler;
    scheduler.reset(new PNDMScheduler);
    mAlphas = scheduler->get_alphas();
}

bool StableDiffusion::load() {
    AUTOTIME;
    ScheduleConfig config;
    BackendConfig backendConfig;
    config.type = mBackendType;
    if(config.type == MNN_FORWARD_CPU) {
        config.numThread = 4;
    } else if(config.type == MNN_FORWARD_OPENCL) {
        config.mode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST;
    } else {
        config.numThread = 1;
    }
    backendConfig.memory = BackendConfig::Memory_Low;
    backendConfig.precision = BackendConfig::Precision_Low;
    config.backendConfig = &backendConfig;
    
    auto exe = ExecutorScope::Current();
    exe->lazyEval = false;
    exe->setGlobalExecutorConfig(config.type, backendConfig, config.numThread);
    
    Module::Config module_config;
    module_config.shapeMutable = false;
    // module_config.rearrange = true;
    runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
    
    if (config.type == MNN_FORWARD_OPENCL) {
        const char* cacheFileName = ".tempcache";
        runtime_manager_->setCache(cacheFileName);
    }
    // need to consider memory
    if(mMemoryMode == 0) {
        runtime_manager_->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
    } else if(mMemoryMode == 2) {
        runtime_manager_->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 1);
    }
    if(config.type == MNN_FORWARD_CPU) {
        runtime_manager_->setHint(Interpreter::DYNAMIC_QUANT_OPTIONS, 2);
    }
    mLatentVar = _Input({1, 4, 64, 64}, NCHW, halide_type_of<float>());
    mPromptVar = _Input({2, mMaxTextLen}, NCHW, halide_type_of<int>());
    mTimestepVar = _Input({1}, NCHW, halide_type_of<int>());
    mLatentVar->writeMap<int8_t>();
    mPromptVar->writeMap<int8_t>();
    mTimestepVar->writeMap<int8_t>();
    mSampleVar = _Concat({mLatentVar, mLatentVar}, 0);
    
    if(mMemoryMode > 0) {
        MNN_PRINT("First time initilizing may cost a few seconds to create cachefile, please wait ...\n");
    }
    
    mModules.resize(3);
    // load text_encoder model (always loaded upfront)
    {
        std::string model_path = mModelPath + "/text_encoder.mnn";
        mModules[0].reset(Module::load(
                                       {"input_ids"}, {"last_hidden_state", "pooler_output"}, model_path.c_str(), runtime_manager_, &module_config));
    }
    // load unet and vae_decoder only if not in low memory mode
    if (mMemoryMode != 0) {
        std::string unet_path = mModelPath + "/unet.mnn";
        mModules[1].reset(Module::load(
                                       {"sample", "timestep", "encoder_hidden_states"}, {"out_sample"}, unet_path.c_str(), runtime_manager_, &module_config));
        std::string vae_path = mModelPath + "/vae_decoder.mnn";
        mModules[2].reset(Module::load(
                                       {"latent_sample"}, {"sample"}, vae_path.c_str(), runtime_manager_, &module_config));
    } else {
        MNN_PRINT("[SD] Low memory mode: UNet and VAE decoder will be loaded on demand\n");
    }
    
    // tokenizer loading
    if(mModelType == STABLE_DIFFUSION_1_5) {
        mTokenizer.reset(new CLIPTokenizer);
    } else if(mModelType == STABLE_DIFFUSION_TAIYI_CHINESE) {
        mTokenizer.reset(new BertTokenizer);
    }
    mTokenizer->load(mModelPath);
    
    // Resize fix for loaded modules
    for (auto& m : mModules) {
        if (m) m->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
    }
    
    return true;
}

VARP StableDiffusion::text_encoder(const std::vector<int>& ids) {
    AUTOTIME;
    
    memcpy((void *)mPromptVar->writeMap<int8_t>(), ids.data(), 2*mMaxTextLen*sizeof(int));
    
    auto outputs = mModules[0]->onForward({mPromptVar});
    auto output = _Convert(outputs[0], NCHW);
    output.fix(VARP::CONSTANT);
    return output;
}

VARP StableDiffusion::step_plms(VARP sample, VARP model_output, int index) {
    int timestep = mTimeSteps[index];
    int prev_timestep = 0;
    if (index + 1 < mTimeSteps.size()) {
        prev_timestep = mTimeSteps[index + 1];
    }
    if (index != 1) {
        if (mEts.size() >= 4) {
            mEts[mEts.size() - 4] = nullptr;
        }
        mEts.push_back(model_output);
    } else {
        timestep = mTimeSteps[0];
        prev_timestep = mTimeSteps[1];
    }
    int ets = mEts.size() - 1;
    if (index == 0) {
        mSample = sample;
    } else if (index == 1) {
        model_output = (model_output + mEts[ets]) * _Const(0.5);
        sample = mSample;
    } else if (ets == 1) {
        model_output = (_Const(3.0) * mEts[ets] - mEts[ets-1]) * _Const(0.5);
    } else if (ets == 2) {
        model_output = (_Const(23.0) * mEts[ets] - _Const(16.0) * mEts[ets-1] + _Const(5.0) * mEts[ets-2]) * _Const(1.0 / 12.0);
    } else if (ets >= 3) {
        model_output = _Const(1. / 24.) * (_Const(55.0) * mEts[ets] - _Const(59.0) * mEts[ets-1] + _Const(37.0) * mEts[ets-2] - _Const(9.0) * mEts[ets-3]);
    }
    auto alpha_prod_t = mAlphas[timestep];
    auto alpha_prod_t_prev = mAlphas[prev_timestep];
    auto beta_prod_t = 1 - alpha_prod_t;
    auto beta_prod_t_prev = 1 - alpha_prod_t_prev;
    auto sample_coeff = std::sqrt(alpha_prod_t_prev / alpha_prod_t);
    auto model_output_denom_coeff = alpha_prod_t * std::sqrt(beta_prod_t_prev) + std::sqrt(alpha_prod_t * beta_prod_t * alpha_prod_t_prev);
    auto prev_sample = _Scalar(sample_coeff) * sample - _Scalar((alpha_prod_t_prev - alpha_prod_t)/model_output_denom_coeff) * model_output;
    return prev_sample;
}

VARP StableDiffusion::unet(VARP text_embeddings, int iterNum, int randomSeed, std::function<void(int)> progressCallback) {
    if(mMemoryMode != 1) {
        // Copy text_embeddings to independent CPU tensor before freeing text encoder
        auto info = text_embeddings->getInfo();
        if (info) {
            size_t n = 1; for (auto d : info->dim) n *= d;
            std::vector<float> buf(n);
            const float* src = text_embeddings->readMap<float>();
            if (src) memcpy(buf.data(), src, n * sizeof(float));
            VARP tmp = _Input(info->dim, info->order, halide_type_of<float>());
            memcpy(tmp->writeMap<float>(), buf.data(), n * sizeof(float));
            tmp.fix(VARP::CONSTANT);
            text_embeddings = tmp;
        }
        mModules[0].reset();
        MNN_PRINT("[SD] Text encoder unloaded\n");
    }
    // Lazy load UNet on demand
    if (!mModules[1]) {
        std::string unet_path = mModelPath + "/unet.mnn";
        MNN_PRINT("[SD] Load UNet on demand: %s\n", unet_path.c_str());
        Module::Config mc; mc.shapeMutable = false;
        mModules[1].reset(Module::load(
            {"sample", "timestep", "encoder_hidden_states"}, {"out_sample"}, unet_path.c_str(), runtime_manager_, &mc));
        if (mModules[1]) mModules[1]->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
        if (!mModules[1]) { MNN_PRINT("[SD] Failed to load UNet\n"); return nullptr; }
    }
    if(mInitNoise.size() != 16384) {
        mInitNoise.resize(16384);
    }
#ifdef MNN_DUMP_DATA
    std::ostringstream fileName;
    fileName << "random.txt";
    std::ifstream input(fileName.str().c_str());
    for (int i = 0; i < 16384; ++i) {
        input >> mInitNoise[i];
    }
#else
    int seed = randomSeed < 0 ? std::random_device()() : randomSeed;
    std::mt19937 rng;
    rng.seed(seed);
    
    std::normal_distribution<float> normal(0, 1);
    for (int i = 0; i < 16384; i++) {
        mInitNoise[i] = normal(rng);
    }
#endif
    
    memcpy((void *)mLatentVar->writeMap<int8_t>(), mInitNoise.data(), 16384*sizeof(float));
    
    VARP scalevar = _Input({1}, NCHW, halide_type_of<float>());
    auto scaleptr = scalevar->writeMap<float>();
    scaleptr[0] = 7.5;
    
    
    auto floatVar = _Input({1}, NCHW, halide_type_of<float>());
    auto ptr = floatVar->writeMap<float>();
    auto plms = mLatentVar;
    
    for (int i = 0; i < mTimeSteps.size(); i++) {
        AUTOTIME;
        
        int timestep = mTimeSteps[i];
        ptr[0] = timestep;
        auto temp = _Cast(floatVar, halide_type_of<int>());
        mTimestepVar->input(temp);

        mSampleVar = _Concat({plms, plms}, 0);
        auto outputs = mModules[1]->onForward({mSampleVar, mTimestepVar, text_embeddings});
        auto output = _Convert(outputs[0], NCHW);
        
        auto noise_pred = output;
        
        auto splitvar = _Split(noise_pred, {2}, 0);
        auto noise_pred_uncond = splitvar[0];
        auto noise_pred_text = splitvar[1];
        
        noise_pred = scalevar * (noise_pred_text - noise_pred_uncond) + noise_pred_uncond;
        
        plms = step_plms(plms, noise_pred, i);
        
        if (progressCallback) {
            progressCallback((2 + i) * 100 / (iterNum + 3)); // percent
        }
    }
    plms.fix(VARP::CONSTANT);
    return plms;
}

VARP StableDiffusion::vae_decoder(VARP latent) {
    if(mMemoryMode != 1) {
        mModules[1].reset();
        MNN_PRINT("[SD] UNet unloaded\n");
    }
    // Lazy load VAE decoder on demand
    if (!mModules[2]) {
        std::string vae_path = mModelPath + "/vae_decoder.mnn";
        MNN_PRINT("[SD] Load VAE decoder on demand: %s\n", vae_path.c_str());
        Module::Config mc; mc.shapeMutable = false;
        mModules[2].reset(Module::load(
            {"latent_sample"}, {"sample"}, vae_path.c_str(), runtime_manager_, &mc));
        if (mModules[2]) mModules[2]->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
        if (!mModules[2]) { MNN_PRINT("[SD] Failed to load VAE decoder\n"); return nullptr; }
    }
    latent = latent * _Const(1 / 0.18215);
    
    AUTOTIME;
    auto outputs = mModules[2]->onForward({latent});
    auto output = _Convert(outputs[0], NCHW);
    
    auto image = output;
    image = _Relu6(image * _Const(0.5) + _Const(0.5), 0, 1);
    image = _Squeeze(_Transpose(image, {0, 2, 3, 1}));
    image = _Cast(_Round(image * _Const(255.0)), halide_type_of<uint8_t>());
    image = cvtColor(image, COLOR_BGR2RGB);
    image.fix(VARP::CONSTANT);
    return image;
}

bool StableDiffusion::run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed, std::function<void(int)> progressCallback) {
    AUTOTIME;
    mEts.clear();
 
    if(iterNum > 50) {
        iterNum = 50;
        MNN_PRINT("too much number of iterations, iterations will be set to 50.\n");
    }
    if(iterNum < 1) {
        iterNum = 10;
        MNN_PRINT("illegal number of iterations, iterations will be set to 10.\n");
    }
    mTimeSteps.resize(iterNum);
    int step = 1000 / iterNum;
    for(int i = iterNum - 1; i >= 0; i--) {
        mTimeSteps[i] = 1 + (iterNum - 1 - i) * step;
    }

    auto ids = mTokenizer->encode(prompt, mMaxTextLen);

    auto text_embeddings = text_encoder(ids);
    
    if (progressCallback) {
        progressCallback(1 * 100 / (iterNum + 3)); // percent
    }
    auto latent = unet(text_embeddings, iterNum, randomSeed, progressCallback);
    
    auto image = vae_decoder(latent);
    bool res = imwrite(imagePath, image);
    if (res) {
        MNN_PRINT("SUCCESS! write generated image to %s\n", imagePath.c_str());
    }

    if(mMemoryMode != 1) {
        mModules[2].reset();
        MNN_PRINT("[SD] VAE decoder unloaded\n");
    }
    
    if (progressCallback) {
        progressCallback(100); // percent
    }
    return true;
}


// 统一的生成接口实现
// 注意：Stable Diffusion当前实现仅支持text2img模式和512x512分辨率
// input_embeds应该是已经tokenized的文本ids（shape: [2, max_text_len]）
bool StableDiffusion::run(const VARP input_embeds, 
                         const std::string& mode,
                         const std::string& inputImagePath,
                         const std::string& outputImagePath,
                         int width,
                         int height,
                         int iterNum,
                         int randomSeed,
                         bool use_cfg,
                         float cfg_scale,
                         std::function<void(int)> progressCallback) {
    
    MNN_PRINT("Error: stable diffusion model does not support feature vector input.\n");
    return false;

}

} // namespace DIFFUSION
} // namespace MNN
