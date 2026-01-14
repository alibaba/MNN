//
//  model_runner.cpp
//
//  Created by MNN on 2024/01/01.
//  ModelRunner class implementation
//

#include "model_runner.hpp"
#include "log_utils.hpp"
#include "../../../transformers/llm/engine/include/llm/llm.hpp"
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/HalideRuntime.h>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <thread>
#include <future>
#include <string>

#ifdef __ANDROID__
#include "AndroidVideoDecoder.hpp"
#endif

using namespace MNN::Transformer;

ModelRunner::ModelRunner(Llm* llm) : llm_(llm) {
    if (!llm_) {
        throw std::invalid_argument("LLM pointer cannot be null");
    }
}

int ModelRunner::EvalPrompts(const std::vector<std::string>& prompts) {
    int prompt_len = 0;
    int decode_len = 0;
    int64_t vision_time = 0;
    int64_t audio_time = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    int64_t sample_time = 0;
    
    for (int i = 0; i < prompts.size(); i++) {
        const auto& prompt = prompts[i];
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        
        ProcessPrompt(prompt, &std::cout);
        auto context = llm_->getContext();
        prompt_len += context->prompt_len;
        decode_len += context->gen_seq_len;
        vision_time += context->vision_us;
        audio_time += context->audio_us;
        prefill_time += context->prefill_us;
        decode_time += context->decode_us;
        sample_time += context->sample_us;
    }
    
    ShowPerformanceMetrics(prompt_len, decode_len, vision_time, audio_time, 
                          prefill_time, decode_time, sample_time);
    
    return 0;
}

int ModelRunner::EvalFile(const std::string& prompt_file) {
    std::cout << "Reading prompts from: " << prompt_file << "\n";
    
    auto prompts = ReadPromptsFromFile(prompt_file);
    if (prompts.empty()) {
        std::cerr << "Error: Prompt file is empty or could not be read" << std::endl;
        return 1;
    }
    
    return EvalPrompts(prompts);
}

void ModelRunner::InteractiveChat() {
    std::cout << "🚀 Starting interactive chat mode...\n";
    std::cout << "Commands: /help, /reset, /config, /exit\n";
#ifdef LLM_SUPPORT_VISION
#ifndef OPENCV_NOT_AVAILABLE
    std::cout << "💡 You can also use video prompts: <video>path/to/video.mp4</video>\n";
#else
    std::cout << "⚠️  Video processing is disabled (OpenCV not available)\n";
#endif
#endif
    std::cout << "\n";
    
    while (true) {
        std::cout << "👤 User: ";
        std::string input;
        if (!std::getline(std::cin, input)) {
            std::cout << "\n";
            break;
        }
        
        if (input == "/exit") break;
        if (input == "/help") ShowChatHelp();
        else if (input == "/reset") ResetConversation();
        else if (input == "/config") ShowConfig();
        else if (!input.empty()) {
            std::cout << "\n🤖 Assistant: " << std::flush;
            // Use ProcessPrompt to handle both text and video prompts
            ProcessPrompt(input, &std::cout);
            std::cout << "\n";
        }
    }
}

int ModelRunner::ProcessPrompt(const std::string& prompt, std::ostream* output, int max_new_tokens) {
    if (output == nullptr) {
        output = &std::cout;
    }

#ifdef LLM_SUPPORT_VISION
#ifndef OPENCV_NOT_AVAILABLE
    // Check if prompt contains video tags
    std::regex video_regex("<video>(.*?)</video>");
    std::smatch match;
    if (std::regex_search(prompt, match, video_regex)) {
        return ProcessVideoPrompt(prompt, output);
    }
#endif
#endif
    llm_->response(prompt, output, nullptr, max_new_tokens);
    return 0;
}

#ifdef LLM_SUPPORT_VISION
#ifndef OPENCV_NOT_AVAILABLE
MNN::Express::VARP ModelRunner::MatToVar(const cv::Mat& mat) {
    // Ensure the mat is not empty
    if (mat.empty()) {
        MNN_ERROR("Input cv::Mat is empty!\n");
        return nullptr;
    }
    // Only support CV_8UC3 for now
    if (mat.type() != CV_8UC3) {
        MNN_ERROR("Only support CV_8UC3 for MatToVar!\n");
        return nullptr;
    }
    auto var = MNN::Express::_Input({mat.rows, mat.cols, 3}, MNN::Express::NHWC, halide_type_of<uint8_t>());
    auto ptr = var->writeMap<uint8_t>();
    memcpy(ptr, mat.data, mat.total() * mat.elemSize());
    return var;
}

int ModelRunner::ProcessVideoPrompt(const std::string& prompt_str, std::ostream* output) {
    std::regex video_regex("<video>(.*?)</video>");
    std::smatch match;
    if (!std::regex_search(prompt_str, match, video_regex)) {
        return ProcessPrompt(prompt_str, output);
    }

    std::string video_path = match[1].str();
    std::string text_part = std::regex_replace(prompt_str, video_regex, "");
    std::vector<MNN::Express::VARP> images;
    std::string final_prompt = text_part;

#ifdef __ANDROID__
    // Android native decoding path
    LOG_DEBUG("Using Android native decoder for: " + video_path);

    MNN::AndroidVideoDecoder decoder;
    if (!decoder.init(video_path)) {
        LOG_ERROR("Failed to initialize AndroidVideoDecoder for path: " + video_path);
        return 1;
    }

    double fps = decoder.get_fps();
    if (fps <= 0) fps = 30.0; // Fallback fps
    int sample_rate = 2; // frames per second
    int frame_interval = static_cast<int>(fps / sample_rate);
    if (frame_interval <= 0) frame_interval = 1;

            LOG_DEBUG("Sampling video at " + std::to_string(sample_rate) + " fps, interval: " + std::to_string(frame_interval));

    int frame_idx = 0;
    int frames_processed = 0;
    while (frames_processed < 100 && !decoder.is_eos()) {
        cv::Mat frame = decoder.decode_one_frame();
        if (frame.empty()) {
            if (decoder.is_eos()) break;
            continue;
        }

        if (frame_idx % frame_interval == 0) {
            int current_second = static_cast<int>(frame_idx / fps);
            char timestamp[32];
            snprintf(timestamp, sizeof(timestamp), "Frame at %02d:%02d: ", current_second / 60, current_second % 60);
            final_prompt += timestamp;
            final_prompt += "<img></img>";
            
            auto var = MatToVar(frame);
            if (var.get() != nullptr) {
                images.push_back(var);
                frames_processed++;
                LOG_DEBUG("Successfully processed frame " + std::to_string(frame_idx));
            } else {
                LOG_DEBUG("Failed to convert frame " + std::to_string(frame_idx) + " to VARP");
            }
        }
        frame_idx++;
    }
    decoder.release();
    LOG_DEBUG("Android native decoding finished. Processed " + std::to_string(frames_processed) + " frames.");

#else
    // Original OpenCV path for non-Android platforms
    LOG_DEBUG("Using OpenCV decoder for: " + video_path);

    // Check if video file exists
    std::ifstream file_check(video_path);
    if (!file_check.good()) {
        LOG_ERROR("Video file does not exist or is not accessible: " + video_path);
        return 1;
    }
    file_check.close();
    
    LOG_DEBUG("Video file exists and is accessible");
    
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        LOG_ERROR("Failed to open video file with OpenCV: " + video_path);
        return 1;
    }
    
    LOG_DEBUG("Video file opened successfully with OpenCV");
    
    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
    
    LOG_DEBUG("Video properties - FPS: " + std::to_string(fps) + ", Frame count: " + std::to_string(frame_count));
    
    if (fps <= 0) {
        LOG_ERROR("Invalid FPS: " + std::to_string(fps));
        cap.release();
        return 1;
    }
    
    bool frame_count_reliable = (frame_count > 0);
    if (!frame_count_reliable) {
        LOG_DEBUG("Frame count unreliable, will process frames dynamically");
    }
    
    int sample_rate = 2; // frames per second
            LOG_DEBUG("Sample rate: " + std::to_string(sample_rate) + " frames per second");
    
    int frame_interval = static_cast<int>(fps / sample_rate);
    if (frame_interval <= 0) {
        frame_interval = 1;
        LOG_DEBUG("Frame interval calculation failed, using fallback value: " + std::to_string(frame_interval));
    }
    
    LOG_DEBUG("Frame sampling interval: " + std::to_string(frame_interval));
    LOG_DEBUG("Starting frame reading loop...");

    int frames_read = 0;
    int frames_processed = 0;
    
    int max_frames_to_process = frame_count_reliable ? frame_count : 10000;
    
    for (int i = 0; i < max_frames_to_process; ++i) {
        auto future = std::async(std::launch::async, [&]() {
            return cap.grab();
        });

        if (future.wait_for(std::chrono::milliseconds(500)) == std::future_status::timeout) {
            LOG_DEBUG("cap.grab() timed out, assuming end of video.");
            break;
        }

        if (!future.get()) {
            LOG_DEBUG("cap.grab() returned false, assuming end of video.");
            break;
        }

        frames_read++;

        if (i % frame_interval == 0) {
            cv::Mat frame;
            if (!cap.retrieve(frame) || frame.empty()) {
                LOG_DEBUG("Failed to retrieve frame " + std::to_string(i));
                continue;
            }

            int current_second = static_cast<int>(i / fps);
            char timestamp[16];
            snprintf(timestamp, sizeof(timestamp), "Frame at %02d:%02d: ", current_second / 60, current_second % 60);
            final_prompt += timestamp;
            final_prompt += "<img></img>";
            
            auto var = MatToVar(frame);
            if (var.get() != nullptr) {
                images.push_back(var);
                frames_processed++;
                LOG_DEBUG("Successfully processed frame " + std::to_string(i) + " at second " + std::to_string(current_second));
            } else {
                LOG_DEBUG("Failed to convert frame " + std::to_string(i) + " to VARP");
            }
        }
        
        if (frames_processed >= 100) {
            LOG_DEBUG("Reached maximum frame processing limit (100 frames)");
            break;
        }
    }
    
    cap.release();
    LOG_DEBUG("Frame reading completed");
#endif // __ANDROID__

    // --- Common code for both platforms ---
    LOG_DEBUG("Total frames processed: " + std::to_string(images.size()));
    LOG_DEBUG("Final prompt: " + final_prompt);
    std::cout << "Read " << images.size() << " frames from video." << std::endl;
    
    if (images.empty()) {
        LOG_WARNING("No frames were successfully processed from the video, falling back to text-only.");
        return ProcessPrompt(text_part, output);
    }
    
    MNN::AutoTime _t(0, "responseWithImages");
    
    MNN::Transformer::MultimodalPrompt multimodal_prompt;
    multimodal_prompt.prompt_template = final_prompt;
    
    for (size_t i = 0; i < images.size(); ++i) {
        std::string image_key = "img_" + std::to_string(i);
        MNN::Transformer::PromptImagePart image_part;
        image_part.image_data = images[i];
        image_part.width = 0;
        image_part.height = 0;
        multimodal_prompt.images[image_key] = image_part;
    }
    
    LOG_DEBUG("Sending multimodal prompt with " + std::to_string(images.size()) + " images to LLM");
    
    llm_->response(multimodal_prompt, output, nullptr, 9999);
    
    return 0;
}
#endif // OPENCV_NOT_AVAILABLE
#endif // LLM_SUPPORT_VISION

#ifndef LLM_SUPPORT_VISION
// Stubs for when vision is not supported
MNN::Express::VARP ModelRunner::MatToVar(void* mat) {
    std::cerr << "Error: OpenCV not available, video processing disabled" << std::endl;
    return nullptr;
}

int ModelRunner::ProcessVideoPrompt(const std::string& prompt_str, std::ostream* output) {
    std::cerr << "Error: OpenCV not available, video processing disabled" << std::endl;
    return 1;
}
#endif

std::vector<std::string> ModelRunner::ReadPromptsFromFile(const std::string& prompt_file) {
    std::ifstream prompt_fs(prompt_file);
    if (!prompt_fs.is_open()) {
        std::cerr << "Error: Failed to open prompt file: " << prompt_file << std::endl;
        return {};
    }
    
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.back() == '\r') {
            prompt.pop_back();
        }
        if (!prompt.empty()) {
            prompts.push_back(prompt);
        }
    }
    prompt_fs.close();
    
    return prompts;
}

void ModelRunner::ShowPerformanceMetrics(int prompt_len, int decode_len, 
                                       int64_t vision_time, int64_t audio_time,
                                       int64_t prefill_time, int64_t decode_time, 
                                       int64_t sample_time) {
    float vision_s = vision_time / 1e6;
    float audio_s = audio_time / 1e6;
    float prefill_s = (prefill_time / 1e6 + vision_s + audio_s);
    float decode_s = decode_time / 1e6;
    float sample_s = sample_time / 1e6;
    
    std::cout << "\n#################################\n";
    std::cout << "prompt tokens num = " << prompt_len << "\n";
    std::cout << "decode tokens num = " << decode_len << "\n";
    std::cout << " vision time = " << std::fixed << std::setprecision(2) << vision_s << " s\n";
    std::cout << "  audio time = " << std::fixed << std::setprecision(2) << audio_s << " s\n";
    std::cout << "prefill time = " << std::fixed << std::setprecision(2) << prefill_s << " s\n";
    std::cout << " decode time = " << std::fixed << std::setprecision(2) << decode_s << " s\n";
    std::cout << " sample time = " << std::fixed << std::setprecision(2) << sample_s << " s\n";
    std::cout << "prefill speed = " << std::fixed << std::setprecision(2) << (prompt_len / prefill_s) << " tok/s\n";
    std::cout << " decode speed = " << std::fixed << std::setprecision(2) << (decode_len / decode_s) << " tok/s\n";
    std::cout << "##################################\n";
}

void ModelRunner::ShowChatHelp() {
    std::cout << "\nAvailable commands:\n";
    std::cout << "  /help   - Show this help message\n";
    std::cout << "  /reset  - Reset conversation context\n";
    std::cout << "  /config - Show current configuration\n";
    std::cout << "  /exit   - Exit chat mode\n";
#ifdef LLM_SUPPORT_VISION
#ifndef OPENCV_NOT_AVAILABLE
    std::cout << "\nVideo prompts:\n";
    std::cout << "  Use <video>path/to/video.mp4</video> in your message to process video files\n";
    std::cout << "  Example: \"What's happening in this video? <video>demo.mp4</video>\"\n";
#endif
#endif
    std::cout << "\n";
}

void ModelRunner::ResetConversation() {
    llm_->reset();
    std::cout << "🔄 Conversation context reset.\n\n";
}

void ModelRunner::ShowConfig() {
    std::cout << "Current configuration:\n";
    std::cout << "  - LLM model loaded successfully\n";
    std::cout << "  - Vision support: ";
#ifdef LLM_SUPPORT_VISION
#ifndef OPENCV_NOT_AVAILABLE
    std::cout << "enabled (OpenCV available)\n";
#else
    std::cout << "enabled (OpenCV not available)\n";
#endif
#else
    std::cout << "disabled\n";
#endif
    std::cout << "  - Audio support: enabled\n";
    std::cout << "\n";
}
