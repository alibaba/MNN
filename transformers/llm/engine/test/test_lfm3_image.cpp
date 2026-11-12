#include <MNN/expr/ExprCreator.hpp>
#include <llm/llm.hpp>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <memory>
#include <sstream>

using namespace MNN::Express;
using namespace MNN::Transformer;

static VARP image(int height, int width, bool red) {
    auto result = _Input({height, width, 3}, NHWC, halide_type_of<uint8_t>());
    auto pixels = result->writeMap<uint8_t>();
    for (int i = 0; i < height * width; ++i) {
        pixels[3 * i] = red ? 0 : 255;
        pixels[3 * i + 1] = 0;
        pixels[3 * i + 2] = red ? 255 : 0;
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: test_lfm3_image LFM3-VL-config.json\n");
        return 2;
    }
    std::unique_ptr<Llm> model(Llm::createLLM(argv[1]));
    if (!model || !model->load())
        return 3;
    model->set_config("{\"jinja\":{\"context\":{\"enable_thinking\":false}},\"sampler_type\":\"greedy\"}");
    for (auto size : std::vector<std::pair<int, int>>{
             {16, 16}, {16, 64}, {64, 16}, {31, 31}, {32, 32}, {33, 33}, {64, 96}, {256, 256}}) {
        model->reset();
        MultimodalPrompt prompt;
        prompt.prompt_template = model->apply_chat_template("<img>image</img>What color is this image? Reply briefly.");
        prompt.images["image"] = {image(size.first, size.second, true), size.second, size.first};
        auto ids = model->tokenizer_encode(prompt);
        if (std::count(ids.begin(), ids.end(), 125030) <= 0)
            return 4;
        auto embedding = model->embedding(ids);
        if (!embedding.get() || !embedding->getInfo())
            return 5;
        auto data = embedding->readMap<float>();
        if (!data)
            return 6;
        for (int i = 0; i < embedding->getInfo()->size; ++i)
            if (!std::isfinite(data[i]))
                return 7;
        if (size.first == 16 && size.second == 16) {
            std::ostringstream output;
            model->response(prompt, &output, nullptr, 48);
            if (model->getContext()->status != LlmStatus::NORMAL_FINISHED &&
                model->getContext()->status != LlmStatus::MAX_TOKENS_FINISHED)
                return 8;
            auto text = output.str();
            std::printf("SMALL_IMAGE %s\n", text.c_str());
            if (text.find("red") == std::string::npos && text.find("Red") == std::string::npos)
                return 8;
        }
        std::printf("IMAGE %dx%d finite=true\n", size.first, size.second);
    }
    model->reset();
    MultimodalPrompt pair;
    pair.prompt_template = model->apply_chat_template(
        "<img>red</img><img>blue</img>What are the colors of the first and second images? Reply briefly.");
    pair.images["red"] = {image(64, 96, true), 96, 64};
    pair.images["blue"] = {image(96, 64, false), 64, 96};
    std::ostringstream output;
    model->response(pair, &output, nullptr, 64);
    auto text = output.str();
    std::printf("MULTI_IMAGE %s\n", text.c_str());
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
    if (model->getContext()->status != LlmStatus::NORMAL_FINISHED &&
        model->getContext()->status != LlmStatus::MAX_TOKENS_FINISHED)
        return 9;
    if (text.find("red") == std::string::npos || text.find("blue") == std::string::npos ||
        text.find("red") > text.find("blue"))
        return 9;
    std::unique_ptr<Llm> unloaded(Llm::createLLM(argv[1]));
    if (!unloaded)
        return 10;
    unloaded->reset();
    std::ostringstream unloadedOutput;
    unloaded->response("Hello", &unloadedOutput, nullptr, 8);
    if (unloaded->getContext()->status != LlmStatus::NOT_LOADED || !unloadedOutput.str().empty() ||
        !unloaded->getContext()->output_tokens.empty())
        return 10;
    unloaded.reset();
    model.reset();
    for (bool video : {false, true}) {
        std::unique_ptr<Llm> rejected(Llm::createLLM(argv[1]));
        if (!rejected || !rejected->load())
            return 10;
        rejected->set_config("{\"jinja\":{\"context\":{\"enable_thinking\":false}},\"sampler_type\":\"greedy\"}");
        MultimodalPrompt invalid;
        if (video) {
            invalid.prompt_template = "<video>clip</video>Describe.";
            PromptVideoPart clip;
            clip.width = clip.height = 64;
            clip.frames = {image(64, 64, true), image(64, 64, true)};
            clip.timestamps = {0.0f, 0.5f};
            invalid.videos["clip"] = clip;
        } else {
            invalid.prompt_template = "<img>invalid</img>Describe.";
            invalid.images["invalid"] = {image(1, 201, true), 201, 1};
        }
        for (bool withImage : {false, true}) {
            rejected->reset();
            std::ostringstream empty;
            rejected->response(invalid, &empty, nullptr, 8);
            if (rejected->getContext()->status != LlmStatus::INTERNAL_ERROR || !empty.str().empty() ||
                !rejected->getContext()->output_tokens.empty())
                return 11;
            rejected->reset();
            if (rejected->getContext()->status != LlmStatus::RUNNING)
                return 12;
            MultimodalPrompt valid;
            valid.prompt_template =
                rejected->apply_chat_template(withImage ? "<img>image</img>What color is this image? Reply briefly."
                                                        : "What is the capital of France? Reply briefly.");
            if (withImage)
                valid.images["image"] = {image(32, 32, true), 32, 32};
            std::ostringstream recovered;
            rejected->response(valid, &recovered, nullptr, 48);
            if ((rejected->getContext()->status != LlmStatus::NORMAL_FINISHED &&
                 rejected->getContext()->status != LlmStatus::MAX_TOKENS_FINISHED) ||
                recovered.str().empty() || rejected->getContext()->output_tokens.empty())
                return 13;
            std::unique_ptr<Llm> fresh(Llm::createLLM(argv[1]));
            if (!fresh || !fresh->load())
                return 14;
            fresh->set_config("{\"jinja\":{\"context\":{\"enable_thinking\":false}},\"sampler_type\":\"greedy\"}");
            std::ostringstream baseline;
            fresh->response(valid, &baseline, nullptr, 48);
            if (baseline.str() != recovered.str() || fresh->getContext()->status != rejected->getContext()->status ||
                fresh->getContext()->output_tokens != rejected->getContext()->output_tokens)
                return 15;
            auto answer = recovered.str();
            std::transform(answer.begin(), answer.end(), answer.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (answer.find(withImage ? "red" : "paris") == std::string::npos)
                return 16;
            std::printf("RESET_RECOVERY invalid=%s valid=%s fresh_equal=true\n", video ? "video" : "image",
                        withImage ? "image" : "text");
        }
    }
    std::puts("LFM3 images, unsupported input and reset recovery: PASS");
    return 0;
}
