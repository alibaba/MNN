#include <random>
#include <fstream>
#include <chrono>
#include "pipeline.hpp"
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

//#define TEXT_MAX_LEN 512 // Taiyi_SD
#define TEXT_MAX_LEN 77 // SD_1.5

//#define MNN_DUMP_DATA

using namespace CV;

namespace diffusion {

static inline int64_t getTime() {
    uint64_t time;
#if defined(_MSC_VER)
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    uint64_t sec = now.QuadPart / freq.QuadPart;
    uint64_t usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
    time = sec * 1000000 + usec;
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#endif
    return time;
}

void display_progress(int cur, int total){
    putchar('\r');
    printf("[");
    for (int i = 0; i < cur; i++) putchar('#');
    for (int i = 0; i < total - cur; i++) putchar('-');
    printf("]");
    fprintf(stdout, "  [%3d%%]", cur * 100 / total);
    if (cur == total) putchar('\n');
    fflush(stdout);
}

Pipeline::Pipeline(std::string modelPath) : mModelPath(modelPath) {
    std::ifstream alphaFile(modelPath + "/alphas.txt");
    int index = 0;
    float alpha;
    while (alphaFile >> alpha) {
        mAlphas.push_back(alpha);
    }
#if 0 // Different steps setting
    mTimeSteps = { // 50 steps
        981, 961, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,
        721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461,
        441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181,
        161, 141, 121, 101,  81,  61,  41,  21,   1
    };
   mTimeSteps = { // 20 steps
       951, 901, 901, 851, 801, 751, 701, 651, 601, 551, 501, 451,
       401, 351, 301, 251,  201,  151,  101,  51,   1
   };
#endif
     mTimeSteps = {
         801, 601, 601,
         401, 201,  1
     };
}

bool Pipeline::load_modules(std::string modelPath) {
    AUTOTIME;

    ScheduleConfig config;
    BackendConfig backendConfig;
//    config.type          = MNN_FORWARD_CPU;
    config.type          = MNN_FORWARD_OPENCL;
    config.numThread     = 65;
    backendConfig.precision = BackendConfig::Precision_Normal;
    backendConfig.memory = BackendConfig::Memory_Low;
    config.backendConfig = &backendConfig;


    Module::Config module_config;
    module_config.shapeMutable = false;
    // module_config.rearrange = true;
    runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));

    // if (config.type == MNN_FORWARD_OPENCL) {
    //     const char* cacheFileName = ".tempcache";
    //     runtime_manager_->setCache(cacheFileName);
    // }
    
    mModules.resize(3);
    // load text_encoder model
    {
        std::string model_path = modelPath + "/text_encoder.mnn";
        mModules[0].reset(Module::load(
            {"input_ids"}, {"last_hidden_state", "pooler_output"}, model_path.c_str(), runtime_manager_, &module_config));

    }
    // load unet model
    {
        std::string model_path = modelPath + "/unet.mnn";
        mModules[1].reset(Module::load(
            {"sample", "timestep", "encoder_hidden_states"}, {"out_sample"}, model_path.c_str(), runtime_manager_, &module_config));
    }
    // load vae_decoder model
    {
        std::string model_path = modelPath + "/vae_decoder.mnn";
        mModules[2].reset(Module::load(
            {"latent_sample"}, {"sample"}, model_path.c_str(), runtime_manager_, &module_config));
    }

    auto exe = ExecutorScope::Current();
    exe->lazyEval = false;
    exe->setGlobalExecutorConfig(config.type, backendConfig, config.numThread);

    return true;
}

VARP Pipeline::text_encoder(const std::vector<int>& ids) {
    auto inputs_ids_ = _Const(ids.data(), {2, TEXT_MAX_LEN}, NCHW, halide_type_of<int>());
    
    auto outputs = mModules[0]->onForward({inputs_ids_});
    auto output = _Convert(outputs[0], NCHW);
    output.fix(VARP::CONSTANT);
    
#ifdef MNN_DUMP_DATA
    auto xx = output->readMap<float>();
    for(int i=0; i<10; i+=2) {
        printf("%f %f ", xx[i], xx[i+TEXT_MAX_LEN*768]);
    }
    printf("\n\n");
#endif
    return output;
}

VARP Pipeline::step_plms(VARP sample, VARP model_output, int index) {
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

VARP Pipeline::unet(VARP text_embeddings) {
    mModules[0].reset();
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::normal_distribution<float> normal(0, 1);
    std::vector<float> initVal(16384);

#ifdef MNN_DUMP_DATA
    std::ostringstream fileName;
    fileName << "random.txt";
    std::ifstream input(fileName.str().c_str());
    for (int i = 0; i < 16384; ++i) {
        input >> initVal[i];
    }
#else
    for (int i = 0; i < 16384; i++) {
        initVal[i] = normal(rng);
    }
#endif
    
    // VARP latentvar = _Const(initVal.data(), {1, 4, 64, 64}, NCHW);
    VARP latentvar = _Input({1, 4, 64, 64}, NCHW, halide_type_of<float>());
    memcpy((void *)latentvar->writeMap<int8_t>(), initVal.data(), 16384*sizeof(float));

    VARP scalevar = _Input({1}, NCHW, halide_type_of<float>());
    auto scaleptr = scalevar->writeMap<float>();
    scaleptr[0] = 7.5;

    
    VARP timestepvar = _Input({1}, NCHW, halide_type_of<int>());
    auto floatVar = _Input({1}, NCHW, halide_type_of<float>());
    auto ptr = floatVar->writeMap<float>();

    for (int i = 0; i < mTimeSteps.size(); i++) {
        AUTOTIME;
        display_progress(i, mTimeSteps.size());

        auto t0 = getTime();

        int timestep = mTimeSteps[i];

        ptr[0] = timestep;
        auto temp = _Cast(floatVar, halide_type_of<int>());
        timestepvar->input(temp);

        VARP samplevar = _Concat({latentvar, latentvar}, 0);
        auto outputs = mModules[1]->onForward({samplevar, timestepvar, text_embeddings});

        auto output = _Convert(outputs[0], NCHW);
        auto t1 = getTime();

        auto noise_pred = output;
        
         auto splitvar = _Split(noise_pred, {2}, 0);
         auto noise_pred_uncond = splitvar[0];
         auto noise_pred_text = splitvar[1];

        noise_pred = scalevar * (noise_pred_text - noise_pred_uncond) + noise_pred_uncond;
        auto t2 = getTime();

        latentvar = step_plms(latentvar, noise_pred, i);
        auto t3 = getTime();
        MNN_PRINT("Times: %f %f %f ms\n", (t1-t0)/ 1000.0f, (t2-t1)/ 1000.0f, (t3-t2)/ 1000.0f);
        // latentvar.fix(VARP::CONSTANT);
#ifdef MNN_DUMP_DATA
        auto xx = output->readMap<float>();
        auto yy = samplevar->readMap<float>();
        auto zz = text_embeddings->readMap<float>();

        for(int i=0; i<6; i+=2) {
            printf("%f %f %f ", xx[i], yy[i], zz[i]);
        }
        for(int i=0; i<6; i+=2) {
            printf("%f %f %f ", xx[16384+i], yy[16384+i], zz[TEXT_MAX_LEN*768+i]);
        }
        printf("\n\n");
#endif
    }
    latentvar.fix(VARP::CONSTANT);
    
#ifdef MNN_DUMP_DATA
    auto xx = latentvar->readMap<float>();
    for(int i=0; i<10; i+=2) {
        printf("%f ", xx[i]);
    }
    printf("\n\n");
#endif
    return latentvar;
}

VARP Pipeline::vae_decoder(VARP latent) {
    mModules[1].reset();
    latent = latent * _Const(1 / 0.18215);
    
    auto outputs = mModules[2]->onForward({latent});
    auto output = _Convert(outputs[0], NCHW);

#ifdef MNN_DUMP_DATA
    auto xx = output->readMap<float>();
    for(int i=0; i<320; i+=32) {
        printf("%f ", xx[i]);
    }
    printf("\n\n");
#endif
    
    auto image = output;
    image = _Relu6(image * _Const(0.5) + _Const(0.5), 0, 1);
    image = _Squeeze(_Transpose(image, {0, 2, 3, 1}));
    image = _Cast(_Round(image * _Const(255.0)), halide_type_of<uint8_t>());
    image = cvtColor(image, COLOR_BGR2RGB);
    image.fix(VARP::CONSTANT);
    return image;
}

bool Pipeline::run(const std::string& sentence, const std::string& img_name) {
    std::unique_ptr<diffusion::Tokenizer> tok;
//    tok.reset(new diffusion::BertTokenizer);
    tok.reset(new diffusion::CLIPTokenizer);
    tok->load(mModelPath);
    load_modules(mModelPath);

    AUTOTIME;
    auto ids = tok->encode(sentence, TEXT_MAX_LEN);
    auto text_embeddings = text_encoder(ids);

    auto latent = unet(text_embeddings);
    auto image = vae_decoder(latent);
    bool res = imwrite(img_name, image);
    if (res) {
        printf("SUCCESS! write to %s\n", img_name.c_str());
    }
    // runtime_manager_->updateCache();
    return res;
}

}
