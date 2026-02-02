//
//  sana_diffusion_demo.cpp
//
//  Sana Diffusion 演示程序
//  
//  展示如何使用Sana模型进行文生图和图像编辑：
//  1. 使用Qwen3-0.6B LLM处理文本prompt
//  2. 通过Connector和Projector桥接特征
//  3. 使用DiT Transformer进行去噪生成
//  4. VAE解码得到最终图像
//

#include <iostream>
#include "diffusion/diffusion.hpp"
#include "diffusion/sana_llm.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>

using namespace MNN::DIFFUSION;
using namespace MNN::Express;

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("=====================================================================================================================\n");
        MNN_PRINT("Sana Diffusion Demo - 基于Qwen3-0.6B的高效文生图模型\n");
        MNN_PRINT("=====================================================================================================================\n");
        MNN_PRINT("Usage: ./sana_diffusion_demo <resource_path> <mode> <prompt> [input_image] [output_image] [width] [height] [steps] [seed] [use_cfg] [cfg_scale]\n");
        MNN_PRINT("\n");
        MNN_PRINT("参数说明:\n");
        MNN_PRINT("  resource_path  : 模型资源路径（包含llm/connector/projector/transformer/vae等模型）\n");
        MNN_PRINT("  mode          : 'text2img' 文生图模式, 'img2img' 图像编辑模式\n");
        MNN_PRINT("  prompt        : 文本描述（支持复杂语义，由Qwen3-0.6B处理）\n");
        MNN_PRINT("  input_image   : 输入图像路径（img2img模式必需，text2img模式忽略，默认: \"\"）\n");
        MNN_PRINT("  output_image  : 输出图像路径（默认: sana_out.jpg）\n");
        MNN_PRINT("  width         : 输出图像宽度（默认: 512，必须是32的倍数）\n");
        MNN_PRINT("  height        : 输出图像高度（默认: 512，必须是32的倍数）\n");
        MNN_PRINT("  steps         : 推理步数（默认: 5，通过蒸馏可用较少步数获得高质量）\n");
        MNN_PRINT("  seed          : 随机种子（默认: 42）\n");
        MNN_PRINT("  use_cfg       : 是否使用CFG引导, 0或1（默认: 0）\n");
        MNN_PRINT("  cfg_scale     : CFG引导强度（默认: 4.5，仅use_cfg=1时生效）\n");
        MNN_PRINT("\n");
        MNN_PRINT("示例:\n");
        MNN_PRINT("  文生图(512x512):  ./sana_diffusion_demo models text2img \"一只可爱的猫咪\" \"\" output.jpg 512 512 5 42 1 4.5\n");
        MNN_PRINT("  文生图(1024x1024): ./sana_diffusion_demo models text2img \"一只可爱的猫咪\" \"\" output.jpg 1024 1024 5 42 1 4.5\n");
        MNN_PRINT("  图像编辑:  ./sana_diffusion_demo models img2img \"添加彩虹\" input.jpg output.jpg 512 512 5 42 0 4.5\n");
        MNN_PRINT("=====================================================================================================================\n");
        return 0;
    }

    std::string resource_path = argv[1];
    std::string mode = argv[2];
    std::string prompt = argv[3];
    std::string image_path = (argc > 4) ? argv[4] : "";
    std::string output_name = (argc > 5) ? argv[5] : "sana_out.jpg";
    int width = (argc > 6) ? atoi(argv[6]) : 512;
    int height = (argc > 7) ? atoi(argv[7]) : 512;
    int steps = (argc > 8) ? atoi(argv[8]) : 5;
    int seed = (argc > 9) ? atoi(argv[9]) : 42;
    bool use_cfg = (argc > 10) ? (atoi(argv[10]) != 0) : false;
    float cfg_scale = (argc > 11) ? atof(argv[11]) : 4.5f;
    
    int memory_mode = 2; // standard，0:卸载
    int backend_type = MNN_FORWARD_CPU;
    
    // 验证mode参数
    if (mode != "text2img" && mode != "img2img") {
        MNN_ERROR("Error: mode must be 'text2img' or 'img2img'\n");
        return -1;
    }
    
    // 验证img2img模式需要输入图像
    if (mode == "img2img" && image_path.empty()) {
        MNN_ERROR("Error: img2img mode requires input image path\n");
        return -1;
    }
    
    MNN_PRINT("\n========== 配置信息 ==========\n");
    MNN_PRINT("模式: %s\n", mode.c_str());
    MNN_PRINT("提示词: %s\n", prompt.c_str());
    if (mode == "img2img") {
        MNN_PRINT("输入图像: %s\n", image_path.c_str());
    }
    MNN_PRINT("输出图像: %s\n", output_name.c_str());
    MNN_PRINT("输出分辨率: %dx%d\n", width, height);
    MNN_PRINT("推理步数: %d \n", steps);
    MNN_PRINT("随机种子: %d\n", seed);
    MNN_PRINT("使用CFG: %s\n", use_cfg ? "是" : "否");
    if (use_cfg) {
        MNN_PRINT("CFG强度: %.2f\n", cfg_scale);
    }
    MNN_PRINT("==============================\n\n");

    // ========== 步骤1: 初始化Qwen3-0.6B LLM ==========
    MNN_PRINT("[1/4] 初始化Qwen3-0.6B LLM（文本编码器）...\n");
    std::string llm_path = resource_path + "/llm/";
    SanaLlm sana_llm(llm_path);
    
    // ========== 步骤2: 初始化Diffusion模型 ==========
    MNN_PRINT("[2/4] 初始化Diffusion模型（Connector + Projector + DiT + VAE）...\n");
    std::unique_ptr<Diffusion> diffusion(Diffusion::createDiffusion(
        resource_path, 
        SANA_DIFFUSION, 
        (MNNForwardType)backend_type, 
        memory_mode
    ));
    diffusion->load();
    
    // ========== 步骤3: LLM处理文本 ==========
    MNN_PRINT("[3/4] LLM处理文本prompt...\n");
    VARP llm_out;
    
    if (use_cfg) {
        // CFG模式：同时生成正负样本特征（batch_size=2）
        MNN_PRINT("  CFG模式：生成正负样本特征\n");
        llm_out = sana_llm.process(prompt, true, "");
    } else {
        // 非CFG模式：只生成正样本特征（batch_size=1）
        MNN_PRINT("  非CFG模式：生成单一特征\n");
        llm_out = sana_llm.process(prompt, false);
    }
    
    if (llm_out.get() == nullptr) {
        MNN_ERROR("LLM处理失败\n");
        return -1;
    }
    
    // ========== 步骤4: Diffusion生成图像 ==========
    MNN_PRINT("[4/4] Diffusion生成图像...\n");
    auto progress = [](int p) {
        std::cout << "  生成进度: " << p << "%\r" << std::flush;
    };

    bool success = diffusion->run(
        llm_out,           // LLM特征（来自Qwen3-0.6B）
        mode,              // 模式：text2img或img2img
        image_path,        // 输入图像（img2img模式）
        output_name,       // 输出路径
        width,             // 输出宽度
        height,            // 输出高度
        steps,             // 推理步数（蒸馏加速）
        seed,              // 随机种子
        use_cfg,           // 是否使用CFG
        cfg_scale,         // CFG强度
        progress           // 进度回调
    );
    
    if (success) {
        MNN_PRINT("\n\n========== 生成完成 ==========\n");
        MNN_PRINT("✓ 图像已保存至: %s\n", output_name.c_str());
        MNN_PRINT("==============================\n");
    } else {
        MNN_ERROR("\n生成失败\n");
        return -1;
    }
    
    return 0;
}
