//
//  llm_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "llm/llm.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <fstream>
#include <sstream>
#include <stdlib.h>
using namespace MNN::Transformer;

static void tuning_prepare(Llm* llm) {
    MNN_PRINT("Prepare for tuning opt Begin\n");
    llm->tuning(OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
    MNN_PRINT("Prepare for tuning opt End\n");
}

std::vector<std::vector<std::string>> parse_csv(const std::vector<std::string>& lines) {
    std::vector<std::vector<std::string>> csv_data;
    std::string line;
    std::vector<std::string> row;
    std::string cell;
    bool insideQuotes = false;
    bool startCollecting = false;

    // content to stream
    std::string content = "";
    for (auto line : lines) {
        content = content + line + "\n";
    }
    std::istringstream stream(content);

    while (stream.peek() != EOF) {
        char c = stream.get();
        if (c == '"') {
            if (insideQuotes && stream.peek() == '"') { // quote
                cell += '"';
                stream.get(); // skip quote
            } else {
                insideQuotes = !insideQuotes; // start or end text in quote
            }
            startCollecting = true;
        } else if (c == ',' && !insideQuotes) { // end element, start new element
            row.push_back(cell);
            cell.clear();
            startCollecting = false;
        } else if ((c == '\n' || stream.peek() == EOF) && !insideQuotes) { // end line
            row.push_back(cell);
            csv_data.push_back(row);
            cell.clear();
            row.clear();
            startCollecting = false;
        } else {
            cell += c;
            startCollecting = true;
        }
    }
    return csv_data;
}

static int benchmark(Llm* llm, const std::vector<std::string>& prompts, int max_token_number) {
    int prompt_len = 0;
    int decode_len = 0;
    int64_t vision_time = 0;
    int64_t audio_time = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    // llm->warmup();
    auto& state = llm->getState();
    if (max_token_number > 0) {
        llm->set_config("{\"max_new_tokens\":1}");
    }
    for (int i = 0; i < prompts.size(); i++) {
        const auto& prompt = prompts[i];
        // prompt start with '#' will be ignored
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        if (max_token_number > 0) {
            llm->response(prompt, &std::cout, nullptr, 0);
            while (!llm->stoped() && state.gen_seq_len_ < max_token_number) {
                llm->generate(1);
            }
        } else {
            llm->response(prompt);
        }
        prompt_len += state.prompt_len_;
        decode_len += state.gen_seq_len_;
        vision_time += state.vision_us_;
        audio_time += state.audio_us_;
        prefill_time += state.prefill_us_;
        decode_time += state.decode_us_;
    }
    float vision_s = vision_time / 1e6;
    float audio_s = audio_time / 1e6;
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;
    printf("\n#################################\n");
    printf("prompt tokens num = %d\n", prompt_len);
    printf("decode tokens num = %d\n", decode_len);
    printf(" vision time = %.2f s\n", vision_s);
    printf("  audio time = %.2f s\n", audio_s);
    printf("prefill time = %.2f s\n", prefill_s);
    printf(" decode time = %.2f s\n", decode_s);
    printf("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
    printf(" decode speed = %.2f tok/s\n", decode_len / decode_s);
    printf("##################################\n");
    return 0;
}

static int ceval(Llm* llm, const std::vector<std::string>& lines, std::string filename) {
    auto csv_data = parse_csv(lines);
    int right = 0, wrong = 0;
    std::vector<std::string> answers;
    for (int i = 1; i < csv_data.size(); i++) {
        const auto& elements = csv_data[i];
        std::string prompt = elements[1];
        prompt += "\n\nA. " + elements[2];
        prompt += "\nB. " + elements[3];
        prompt += "\nC. " + elements[4];
        prompt += "\nD. " + elements[5];
        prompt += "\n\n";
        printf("%s", prompt.c_str());
        printf("## 进度: %d / %lu\n", i, lines.size() - 1);
        std::ostringstream lineOs;
        llm->response(prompt.c_str(), &lineOs);
        auto line = lineOs.str();
        printf("%s", line.c_str());
        answers.push_back(line);
    }
    {
        auto position = filename.rfind("/");
        if (position != std::string::npos) {
            filename = filename.substr(position + 1, -1);
        }
        position = filename.find("_val");
        if (position != std::string::npos) {
            filename.replace(position, 4, "_res");
        }
        std::cout << "store to " << filename << std::endl;
    }
    std::ofstream ofp(filename);
    ofp << "id,answer" << std::endl;
    for (int i = 0; i < answers.size(); i++) {
        auto& answer = answers[i];
        ofp << i << ",\""<< answer << "\"" << std::endl;
    }
    ofp.close();
    return 0;
}

static int eval(Llm* llm, std::string prompt_file, int max_token_number) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.back() == '\r') {
            prompt.pop_back();
        }
        prompts.push_back(prompt);
    }
    prompt_fs.close();
    if (prompts.empty()) {
        return 1;
    }
    // ceval
    if (prompts[0] == "id,question,A,B,C,D,answer") {
        return ceval(llm, prompts, prompt_file);
    }
    return benchmark(llm, prompts, max_token_number);
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json <prompt.txt>" << std::endl;
        return 0;
    }
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    MNN::Express::ExecutorScope s(executor);

    std::string config_path = argv[1];
    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    {
        AUTOTIME;
        llm->load();
    }
    if (true) {
        AUTOTIME;
        tuning_prepare(llm.get());
    }
    if (argc < 3) {
        llm->chat();
        return 0;
    }
    int max_token_number = -1;
    if (argc >= 4) {
        std::istringstream os(argv[3]);
        os >> max_token_number;
    }
    std::string prompt_file = argv[2];
    return eval(llm.get(), prompt_file, max_token_number);
}
