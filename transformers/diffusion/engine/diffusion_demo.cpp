#include <iostream>
#include <cstdlib>
#include "diffusion/diffusion.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#ifdef _WIN32
#include <windows.h>
// Convert Windows system encoding (e.g., GBK) to UTF-8
static std::string toUtf8(const char* str) {
    if (!str || !*str) return "";
    // First convert from system codepage to wide string
    int wlen = MultiByteToWideChar(CP_ACP, 0, str, -1, nullptr, 0);
    if (wlen <= 0) return str;
    std::wstring wstr(wlen, 0);
    MultiByteToWideChar(CP_ACP, 0, str, -1, &wstr[0], wlen);
    // Then convert from wide string to UTF-8
    int ulen = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (ulen <= 0) return str;
    std::string utf8(ulen, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, &utf8[0], ulen, nullptr, nullptr);
    // Remove trailing null
    if (!utf8.empty() && utf8.back() == '\0') utf8.pop_back();
    return utf8;
}
#else
static std::string toUtf8(const char* str) { return str ? str : ""; }
#endif

using namespace MNN::DIFFUSION;

int main(int argc, const char* argv[]) {
    if (argc < 9) {
        MNN_PRINT("=====================================================================================================================\n");
        MNN_PRINT("Usage: ./diffusion_demo <resource_path> <model_type> <memory_mode> <backend_type> <iteration_num> <random_seed> <output_image_name> [image_size] [cfg_scale] [cfg_mode] [gpu_mem_mode] [precision_mode] [te_on_cpu] [vae_on_cpu] <prompt_text> [input_image]\n");
        MNN_PRINT("\n");
        MNN_PRINT("Required parameters:\n");
        MNN_PRINT("  resource_path    : Path to model directory\n");
        MNN_PRINT("  model_type       : 0=SD1.5, 1=Taiyi, 2=Sana, 3=ZImage, 4=LongCat\n");
        MNN_PRINT("  memory_mode      : 0=low(load/unload on demand), 1=high(keep all in memory), 2=balance\n");
        MNN_PRINT("  backend_type     : 0=CPU, 3=OpenCL, 7=Vulkan, etc.\n");
        MNN_PRINT("  iteration_num    : Numbe of diffusion steps (e.g., 20, 50)\n");
        MNN_PRINT("  random_seed      : Random seed for reproducibility\n");
        MNN_PRINT("  output_image_name: Output image path\n");
        MNN_PRINT("\n");
        MNN_PRINT("Optional parameters (in order):\n");
        MNN_PRINT("  image_size       : Output image size, supports \"512\" (square) or \"512x768\" (width x height)\n");
        MNN_PRINT("                     (default: 512 for SD1.5/Taiyi, 1024 for ZImage/LongCat)\n");
        MNN_PRINT("  cfg_scale        : Classifier-Free Guidance scale (default: 7.5, range: 1.0~15.0)\n");
        MNN_PRINT("  cfg_mode         : CFG sigma range for dual-UNet models (LongCat only)\n");
        MNN_PRINT("                     0=auto(0.1~0.8), 1=wide(0.1~0.9), 2=standard(0.1~0.8),\n");
        MNN_PRINT("                     3=medium(0.15~0.7), 4=narrow(0.2~0.6), 5=minimal(0.25~0.5)\n");
        MNN_PRINT("  gpu_mem_mode     : OpenCL memory mode: 0=auto, 1=buffer, 2=image (Adreno GPU recommended: 2)\n");
        MNN_PRINT("  precision_mode   : 0=auto(ZImage:Normal, LongCat:High), 1=low(FP16), 2=normal(FP32), 3=high(FP32)\n");
        MNN_PRINT("  te_on_cpu        : Text encoder device: 0=same as UNet, 1=force CPU (recommended for large models)\n");
        MNN_PRINT("  vae_on_cpu       : VAE device: 0=same as UNet, 1=force CPU (for GPU memory saving)\n");
        MNN_PRINT("  prompt_text      : Text prompt (can contain spaces)\n");
        MNN_PRINT("  input_image      : Input image path for image-to-image (optional, auto-detected if file exists)\n");
        MNN_PRINT("=====================================================================================================================\n");
        return 0;
    }

    auto resource_path = argv[1];
    auto model_type = (DiffusionModelType)atoi(argv[2]);
    auto memory_mode = atoi(argv[3]);
    auto backend_type = (MNNForwardType)atoi(argv[4]);
    auto iteration_num = atoi(argv[5]);
    auto random_seed = atoi(argv[6]);
    auto img_name = argv[7];

    std::string input_image_path;
    int image_width = 0;
    int image_height = 0;
    float cfgScale = 7.5f;
    DiffusionCFGMode cfgMode = CFG_MODE_AUTO;
    DiffusionGpuMemoryMode gpuMemoryMode = GPU_MEMORY_BUFFER;
    DiffusionPrecisionMode precisionMode = PRECISION_AUTO;
    bool textEncoderOnCPU = false;
    bool vaeOnCPU = false;
    int prompt_start = 8;
    
    // Parse optional parameters in order: [image_size] [cfg_scale] [cfg_mode] [gpu_mem_mode] [precision_mode] [te_on_cpu] [vae_on_cpu] [input_image]
    // 1. image_size (supports both "512" and "512x512" formats)
    if (argc > prompt_start + 1) {
        std::string sizeStr(argv[prompt_start]);
        size_t xPos = sizeStr.find('x');
        if (xPos != std::string::npos) {
            // Format: widthxheight
            char* endptr = nullptr;
            long w = strtol(sizeStr.substr(0, xPos).c_str(), &endptr, 10);
            if (endptr != nullptr && *endptr == '\0' && w > 0) {
                long h = strtol(sizeStr.substr(xPos + 1).c_str(), &endptr, 10);
                if (endptr != nullptr && *endptr == '\0' && h > 0) {
                    image_width = (int)w;
                    image_height = (int)h;
                    prompt_start += 1;
                }
            }
        } else {
            // Format: single number (square image)
            char* endptr = nullptr;
            long v = strtol(argv[prompt_start], &endptr, 10);
            if (endptr != nullptr && *endptr == '\0' && v > 0) {
                image_width = image_height = (int)v;
                prompt_start += 1;
            }
        }
    }
    // 2. cfg_scale
    if (argc > prompt_start + 1) {
        char* endptr = nullptr;
        float v = strtof(argv[prompt_start], &endptr);
        if (endptr != nullptr && *endptr == '\0') {
            cfgScale = v;
            prompt_start += 1;
        }
    }
    // 3. cfg_mode (紧跟 cfg_scale)
    if (argc > prompt_start + 1) {
        char* endptr = nullptr;
        long v = strtol(argv[prompt_start], &endptr, 10);
        if (endptr != nullptr && *endptr == '\0') {
            if (v < 0) v = 0;
            if (v > 5) v = 5;
            cfgMode = (DiffusionCFGMode)v;
            prompt_start += 1;
        }
    }
    // 4. gpu_mem_mode
    if (argc > prompt_start + 1) {
        char* endptr = nullptr;
        long v = strtol(argv[prompt_start], &endptr, 10);
        if (endptr != nullptr && *endptr == '\0') {
            if (v < 0) v = 0;
            if (v > 2) v = 2;
            gpuMemoryMode = (DiffusionGpuMemoryMode)v;
            prompt_start += 1;
        }
    }
    // 5. precision_mode
    if (argc > prompt_start + 1) {
        char* endptr = nullptr;
        long v = strtol(argv[prompt_start], &endptr, 10);
        if (endptr != nullptr && *endptr == '\0') {
            if (v < 0) v = 0;
            if (v > 3) v = 3;
            precisionMode = (DiffusionPrecisionMode)v;
            prompt_start += 1;
        }
    }
    // 6. te_on_cpu
    if (argc > prompt_start + 1) {
        char* endptr = nullptr;
        long v = strtol(argv[prompt_start], &endptr, 10);
        if (endptr != nullptr && *endptr == '\0') {
            textEncoderOnCPU = (v != 0);
            prompt_start += 1;
        }
    }
    // 7. vae_on_cpu
    if (argc > prompt_start + 1) {
        char* endptr = nullptr;
        long v = strtol(argv[prompt_start], &endptr, 10);
        if (endptr != nullptr && *endptr == '\0') {
            vaeOnCPU = (v != 0);
            prompt_start += 1;
        }
    }
    
    // Collect remaining arguments as prompt, but check last arg for input_image path
    // Format: ... <prompt_text> [input_image]
    // input_image is optional and must be a valid file path (contains / or . and exists)
    std::string input_text;
    int prompt_end = argc;
    
    // Check if last argument looks like a file path (for input_image)
    if (argc > prompt_start) {
        const char* lastArg = argv[argc - 1];
        std::string lastArgStr(lastArg);
        // Check if it looks like a file path (contains / or .) and is not just a number
        char* endptr = nullptr;
        strtol(lastArg, &endptr, 10);
        bool isNumber = (endptr != lastArg && *endptr == '\0');
        bool looksLikePath = (lastArgStr.find('/') != std::string::npos || 
                              (lastArgStr.find('.') != std::string::npos && lastArgStr.length() > 4));
        if (!isNumber && looksLikePath) {
            // Check if file exists
            FILE* f = fopen(lastArg, "r");
            if (f) {
                fclose(f);
                input_image_path = lastArg;
                prompt_end = argc - 1;
            }
        }
    }
    
    for (int i = prompt_start; i < prompt_end; ++i) {
        input_text += toUtf8(argv[i]);
        if (i < prompt_end - 1) {
            input_text += " ";
        }
    }
    
    MNN_PRINT("Model resource path: %s\n", resource_path);
    if (model_type == STABLE_DIFFUSION_1_5) {
        MNN_PRINT("Model type is stable diffusion 1.5\n");
    } else if (model_type == STABLE_DIFFUSION_TAIYI_CHINESE) {
        MNN_PRINT("Model type is stable diffusion taiyi chinese version\n");
    } else if (model_type == SANA_DIFFUSION) {
        MNN_PRINT("Model type is Sana diffusion model (Qwen3-0.6B + DiT)\n");
    } else if (model_type == STABLE_DIFFUSION_ZIMAGE) {
        MNN_PRINT("Model type is ZImage diffusion model\n");
    } else if (model_type == LONGCAT_IMAGE_EDIT) {
        MNN_PRINT("Model type is LongCat Image Edit diffusion model\n");
    } else {
        MNN_PRINT("Error: Model type %d not supported, please check\n", (int)model_type);
        return 0;
    }

    // Print configuration
    const char* memoryModeStr = (memory_mode == 0) ? "Low (load/unload on demand)" : 
                                (memory_mode == 1) ? "High (keep all in memory)" : "Balance";
    const char* precisionModeStr = (precisionMode == PRECISION_AUTO) ? "Auto" :
                                   (precisionMode == PRECISION_LOW) ? "Low(FP16)" :
                                   (precisionMode == PRECISION_NORMAL) ? "Normal(FP32)" : "High(FP32)";
    const char* cfgModeStr = (cfgMode == CFG_MODE_AUTO) ? "Auto(0.1~0.8)" :
                             (cfgMode == CFG_MODE_WIDE) ? "Wide(0.1~0.9)" :
                             (cfgMode == CFG_MODE_STANDARD) ? "Standard(0.1~0.8)" :
                             (cfgMode == CFG_MODE_MEDIUM) ? "Medium(0.15~0.7)" :
                             (cfgMode == CFG_MODE_NARROW) ? "Narrow(0.2~0.6)" : "Minimal(0.25~0.5)";
    
    MNN_PRINT("\n=== Configuration ===\n");
    MNN_PRINT("Model path      : %s\n", resource_path);
    MNN_PRINT("Memory mode     : %s\n", memoryModeStr);
    MNN_PRINT("Backend type    : %d\n", (int)backend_type);
    MNN_PRINT("Iteration steps : %d\n", iteration_num);
    MNN_PRINT("Random seed     : %d\n", random_seed);
    MNN_PRINT("Output image    : %s\n", img_name);
    int defaultSize = (model_type <= STABLE_DIFFUSION_TAIYI_CHINESE ? 512 : 1024);
    int finalWidth = image_width > 0 ? image_width : defaultSize;
    int finalHeight = image_height > 0 ? image_height : defaultSize;
    MNN_PRINT("Image size      : %dx%d\n", finalWidth, finalHeight);
    MNN_PRINT("CFG scale       : %.2f\n", cfgScale);
    MNN_PRINT("CFG mode        : %s\n", cfgModeStr);
    MNN_PRINT("GPU memory mode : %d\n", (int)gpuMemoryMode);
    MNN_PRINT("Precision mode  : %s\n", precisionModeStr);
    MNN_PRINT("Text encoder CPU: %s\n", textEncoderOnCPU ? "Yes" : "No");
    MNN_PRINT("VAE on CPU      : %s\n", vaeOnCPU ? "Yes" : "No");
    if (!input_image_path.empty()) {
        MNN_PRINT("Input image     : %s\n", input_image_path.c_str());
    }
    MNN_PRINT("Prompt          : %s\n", input_text.c_str());
    MNN_PRINT("=====================\n\n");

    
    std::unique_ptr<Diffusion> diffusion;
    const int numThreads = 4;
    // Use full factory method with vaeOnCPU and cfgMode
    diffusion.reset(Diffusion::createDiffusion(resource_path, model_type, backend_type, memory_mode, 
                                                finalWidth, finalHeight,  // width, height (can be different)
                                                textEncoderOnCPU, vaeOnCPU, gpuMemoryMode, precisionMode, 
                                                cfgMode, numThreads));

    diffusion->load();
    
    // callback to show progress
    auto progressDisplay = [](int progress) {
        std::cout << "Progress: " << progress << "%" << std::endl;
    };
    diffusion->run(input_text, img_name, iteration_num, random_seed, cfgScale, progressDisplay, input_image_path);
     
    /*
     when need multi text-generation-image:
     if you choose memory lack mode, need diffusion load with each diffusion run.
     if you choose memory enough mode,  just start another diffusion run, only need diffusion load in first time.
     */
    while(0) {
        if(memory_mode != 1) {
            diffusion->load();
        }
        
        diffusion->run("a big horse", "demo_2.jpg", 20, 42, cfgScale, progressDisplay);
    }
    return 0;
}
