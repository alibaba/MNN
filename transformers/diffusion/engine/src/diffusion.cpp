#include <random>
#include <fstream>
#include <chrono>
#include "diffusion/diffusion.hpp"
#include "tokenizer.hpp"
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

//#define MNN_DUMP_DATA

using namespace CV;

namespace MNN {
namespace DIFFUSION {
    
void display_progress(int cur, int total){
    putchar('\r');
    MNN_PRINT("[");
    for (int i = 0; i < cur; i++) putchar('#');
    for (int i = 0; i < total - cur; i++) putchar('-');
    MNN_PRINT("]");
    fprintf(stdout, "  [%3d%%]", cur * 100 / total);
    if (cur == total) putchar('\n');
    fflush(stdout);
}

Diffusion* Diffusion::createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int iterationNum) {
    Diffusion* diffusion = new Diffusion(modelPath, modelType, backendType, memoryMode, iterationNum);
    return diffusion;
}
    
Diffusion::Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int iterationNum) :
mModelPath(modelPath), mModelType(modelType), mBackendType(backendType), mMemoryMode(memoryMode), mIterationNum(iterationNum) {
    if(modelType == STABLE_DIFFUSION_1_5) {
        mMaxTextLen = 77;
    } else if(modelType == STABLE_DIFFUSION_TAIYI_CHINESE) {
        mMaxTextLen = 512;
    }
    std::ifstream alphaFile(modelPath + "/alphas.txt");
    int index = 0;
    float alpha;
    while (alphaFile >> alpha) {
        mAlphas.push_back(alpha);
    }
    if(iterationNum > 50) {
        iterationNum = 50;
        MNN_PRINT("too much number of iterations, iterations will be set to 50.\n");
    }
    if(iterationNum < 1) {
        iterationNum = 10;
        MNN_PRINT("illegal number of iterations, iterations will be set to 10.\n");
    }
    mTimeSteps.resize(iterationNum + 1);
    int step = 1000 / (iterationNum + 1);
    for(int i = iterationNum; i >= 0; i--) {
        mTimeSteps[i] = 1 + (iterationNum - i) * step;
    }
}
    
Diffusion::~Diffusion() {
    mModules.clear();
    runtime_manager_.reset();
}
    
bool Diffusion::load() {
    AUTOTIME;
    ScheduleConfig config;
    BackendConfig backendConfig;
    config.type = mBackendType;
    if(config.type == MNN_FORWARD_CPU) {
        backendConfig.memory = BackendConfig::Memory_Low;
        config.numThread = 4;
    } else if(config.type == MNN_FORWARD_OPENCL) {
        config.mode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST;
    } else {
        config.numThread = 1;
    }
    
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
    
    mLatentVar = _Input({1, 4, 64, 64}, NCHW, halide_type_of<float>());
    mPromptVar = _Input({2, mMaxTextLen}, NCHW, halide_type_of<int>());
    mTimestepVar = _Input({1}, NCHW, halide_type_of<int>());
    mSampleVar = _Concat({mLatentVar, mLatentVar}, 0);
    
    if(mMemoryMode > 0) {
        MNN_PRINT("First time initilizing may cost a few seconds to create cachefile, please wait ...\n");
    }
    
    VARP text_embeddings;
    mModules.resize(3);
    // load text_encoder model
    {
        std::string model_path = mModelPath + "/text_encoder.mnn";
        MNN_PRINT("Load %s\n", model_path.c_str());
        mModules[0].reset(Module::load(
                                       {"input_ids"}, {"last_hidden_state", "pooler_output"}, model_path.c_str(), runtime_manager_, &module_config));
        
        if(mMemoryMode > 0) {
            auto outputs = mModules[0]->onForward({mPromptVar});
            text_embeddings = _Convert(outputs[0], NCHW);
        }
        display_progress(1, 3);
    }
    // load unet model
    {
        std::string model_path = mModelPath + "/unet.mnn";
        MNN_PRINT("Load %s\n", model_path.c_str());
        mModules[1].reset(Module::load(
                                       {"sample", "timestep", "encoder_hidden_states"}, {"out_sample"}, model_path.c_str(), runtime_manager_, &module_config));
        
        if(mMemoryMode > 0) {
            auto outputs = mModules[1]->onForward({mSampleVar, mTimestepVar, text_embeddings});
            auto output = _Convert(outputs[0], NCHW);
        }
        display_progress(2, 3);
    }
    // load vae_decoder model
    {
        std::string model_path = mModelPath + "/vae_decoder.mnn";
        MNN_PRINT("Load %s\n", model_path.c_str());
        mModules[2].reset(Module::load(
                                       {"latent_sample"}, {"sample"}, model_path.c_str(), runtime_manager_, &module_config));
        
        if(mMemoryMode > 0) {
            auto outputs = mModules[2]->onForward({mLatentVar});
            auto output = _Convert(outputs[0], NCHW);
            // sync
            output->readMap<float>();
        }
        display_progress(3, 3);
    }
    
    // tokenizer loading
    if(mModelType == STABLE_DIFFUSION_1_5) {
        mTokenizer.reset(new CLIPTokenizer);
    } else if(mModelType == STABLE_DIFFUSION_TAIYI_CHINESE) {
        mTokenizer.reset(new BertTokenizer);
    }
    mTokenizer->load(mModelPath);
    
    return true;
}

VARP Diffusion::text_encoder(const std::vector<int>& ids) {
    AUTOTIME;
    
    memcpy((void *)mPromptVar->writeMap<int8_t>(), ids.data(), 2*mMaxTextLen*sizeof(int));
    
    auto outputs = mModules[0]->onForward({mPromptVar});
    auto output = _Convert(outputs[0], NCHW);
    output.fix(VARP::CONSTANT);
    
#ifdef MNN_DUMP_DATA
    auto xx = output->readMap<float>();
    for(int i=0; i<10; i+=2) {
        MNN_PRINT("%f %f ", xx[i], xx[i+mMaxTextLen*768]);
    }
    MNN_PRINT("\n\n");
#endif
    return output;
}

VARP Diffusion::step_plms(VARP sample, VARP model_output, int index) {
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

VARP Diffusion::unet(VARP text_embeddings, std::function<void(int)> progressCallback) {
    if(mMemoryMode == 0) {
        mModules[0].reset();
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
    std::mt19937 rng;
    rng.seed(std::random_device()());
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
        //display_progress(i, mTimeSteps.size());
        
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
        
#ifdef MNN_DUMP_DATA
        auto xx = output->readMap<float>();
        auto yy = mSampleVar->readMap<float>();
        auto zz = text_embeddings->readMap<float>();
        auto mm = mTimestepVar->readMap<int>();

        for(int i=0; i<6; i+=2) {
            MNN_PRINT("(0)%f (1)%f (2)%f (3)%d ", xx[i], yy[i], zz[i] ,mm[0]);
        }
        MNN_PRINT("\n");
        for(int i=0; i<6; i+=2) {
            MNN_PRINT("(0)%f (1)%f (2)%f ", xx[16384+i], yy[16384+i], zz[mMaxTextLen*768+i]);
        }
        MNN_PRINT("\n\n");
#endif
        if (progressCallback) {
            progressCallback((2 + i) * 100 / (mIterationNum + 3)); // percent
        }
        
    }
    plms.fix(VARP::CONSTANT);
    
#ifdef MNN_DUMP_DATA
    auto xx = plms->readMap<float>();
    for(int i=0; i<10; i+=2) {
        MNN_PRINT("%f ", xx[i]);
    }
    MNN_PRINT("\n\n");
#endif
    return plms;
}

VARP Diffusion::vae_decoder(VARP latent) {
    if(mMemoryMode == 0) {
        mModules[1].reset();
    }
    latent = latent * _Const(1 / 0.18215);
    
    AUTOTIME;
    auto outputs = mModules[2]->onForward({latent});
    auto output = _Convert(outputs[0], NCHW);
    
#ifdef MNN_DUMP_DATA
    auto xx = output->readMap<float>();
    for(int i=0; i<320; i+=32) {
        MNN_PRINT("%f ", xx[i]);
    }
    MNN_PRINT("\n\n");
#endif
    
    auto image = output;
    image = _Relu6(image * _Const(0.5) + _Const(0.5), 0, 1);
    image = _Squeeze(_Transpose(image, {0, 2, 3, 1}));
    image = _Cast(_Round(image * _Const(255.0)), halide_type_of<uint8_t>());
    image = cvtColor(image, COLOR_BGR2RGB);
    image.fix(VARP::CONSTANT);
    return image;
}

bool Diffusion::run(const std::string prompt, const std::string imagePath, std::function<void(int)> progressCallback) {
    AUTOTIME;
    mEts.clear();
 
    auto ids = mTokenizer->encode(prompt, mMaxTextLen);
    auto text_embeddings = text_encoder(ids);
    
    if (progressCallback) {
        progressCallback(1 * 100 / (mIterationNum + 3)); // percent
    }
    auto latent = unet(text_embeddings, progressCallback);
    
    auto image = vae_decoder(latent);
    bool res = imwrite(imagePath, image);
    if (res) {
        MNN_PRINT("SUCCESS! write generated image to %s\n", imagePath.c_str());
    }

    if(mMemoryMode == 0) {
        mModules[2].reset();
    }
    
    if (progressCallback) {
        progressCallback(100); // percent
    }
    return true;
}
}
}
