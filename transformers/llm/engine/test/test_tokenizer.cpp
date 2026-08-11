//
//  test_tokenizer.cpp
//
//  Tokenizer unit test for PipelineTokenizer (.mtok format).
//  Loads test_cases.jsonl from each model directory, encodes input,
//  and compares with expected ids_raw / decoded_full.
//
//  Usage: test_tokenizer [models_path] [model_filter]
//

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include "tokenizer/tokenizer.hpp"

#include "ujson.hpp"

using namespace MNN::Transformer;

// ANSI colors
static const char* RED    = "\033[31m";
static const char* GREEN  = "\033[32m";
static const char* YELLOW = "\033[33m";
static const char* RESET  = "\033[0m";

static std::vector<std::string> list_dirs(const std::string& path) {
    std::vector<std::string> dirs;
    DIR* d = opendir(path.c_str());
    if (!d) return dirs;
    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        if (entry->d_name[0] == '.') continue;
        std::string full = path + "/" + entry->d_name;
        struct stat st;
        if (stat(full.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            dirs.push_back(entry->d_name);
        }
    }
    closedir(d);
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

static std::string ids_to_str(const std::vector<int>& ids) {
    std::string s = "[";
    for (size_t i = 0; i < ids.size(); i++) {
        if (i) s += ",";
        s += std::to_string(ids[i]);
    }
    s += "]";
    return s;
}

struct TestResult {
    std::string model;
    int total;
    int passed;
    int failed;
    int skipped;
};

static TestResult test_model(const std::string& models_path, const std::string& model_name) {
    TestResult result = {model_name, 0, 0, 0, 0};
    std::string model_dir = models_path + "/" + model_name;

    // Find tokenizer file (.mtok preferred, then .txt)
    std::string tok_path = model_dir + "/tokenizer.mtok";
    {
        struct stat st;
        if (stat(tok_path.c_str(), &st) != 0) {
            tok_path = model_dir + "/tokenizer.txt";
            if (stat(tok_path.c_str(), &st) != 0) {
                std::cerr << YELLOW << "  SKIP: no tokenizer file found" << RESET << std::endl;
                return result;
            }
        }
    }

    // Load tokenizer
    Tokenizer* tok = Tokenizer::createTokenizer(tok_path);
    if (!tok) {
        std::cerr << YELLOW << "  SKIP: failed to load tokenizer" << RESET << std::endl;
        return result;
    }

    // Read test_cases.jsonl
    std::string jsonl_path = model_dir + "/test_cases.jsonl";
    std::ifstream jsonl_file(jsonl_path);
    if (!jsonl_file.is_open()) {
        std::cerr << YELLOW << "  SKIP: no test_cases.jsonl" << RESET << std::endl;
        delete tok;
        return result;
    }

    std::string line;
    int line_no = 0;
    while (std::getline(jsonl_file, line)) {
        line_no++;
        if (line.empty()) continue;

        ujson::json j = ujson::json::parse(line);
        if (j.is_null()) {
            result.skipped++;
            continue;
        }

        std::string type = j.value("type", "");
        if (type == "chat") {
            // Chat template test
            if (tok->chat_template().empty()) {
                result.skipped++;
                continue;
            }
            std::string name = j.value("name", "");
            bool add_gen = j.value("add_generation_prompt", true);
            std::string expected_text = j.value("formatted_text", "");

            // Parse messages — use extra_json for messages with complex fields
            ChatMessages messages;
            if (j.contains("messages") && j["messages"].is_array()) {
                size_t n = j["messages"].size();
                for (size_t i = 0; i < n; i++) {
                    auto msg = j["messages"][i];
                    std::string role = msg.value("role", "");
                    std::string content = msg.value("content", "");
                    bool has_extra = msg.contains("tool_calls") ||
                                     msg.contains("reasoning_content");
                    if (has_extra) {
                        // "json" role signals second is full JSON message object
                        messages.push_back({"json", msg.dump()});
                    } else {
                        messages.push_back({role, content});
                    }
                }
            }
            result.total++;

            std::string actual_text = tok->apply_chat_template(messages, add_gen);
            // Normalize dynamic dates: strftime_now generates current date which varies per run.
            // Replace "Today Date: <actual_date>" with the expected date for stable comparison.
            {
                const std::string prefix = "Today Date: ";
                auto epos = expected_text.find(prefix);
                auto apos = actual_text.find(prefix);
                if (epos != std::string::npos && apos != std::string::npos) {
                    auto eend = expected_text.find('\n', epos);
                    auto aend = actual_text.find('\n', apos);
                    if (eend != std::string::npos && aend != std::string::npos) {
                        actual_text.replace(apos + prefix.size(), aend - apos - prefix.size(),
                                            expected_text.substr(epos + prefix.size(), eend - epos - prefix.size()));
                    }
                }
            }
            if (actual_text == expected_text) {
                result.passed++;
            } else {
                result.failed++;
                std::cerr << RED << "  FAIL chat \"" << name << "\" line " << line_no << RESET << std::endl;
                // Show first difference
                size_t diff_pos = 0;
                while (diff_pos < actual_text.size() && diff_pos < expected_text.size() &&
                       actual_text[diff_pos] == expected_text[diff_pos]) diff_pos++;
                std::cerr << "    diff at pos " << diff_pos << std::endl;
                size_t ctx_start = diff_pos > 20 ? diff_pos - 20 : 0;
                size_t ctx_end_e = std::min(diff_pos + 40, expected_text.size());
                size_t ctx_end_a = std::min(diff_pos + 40, actual_text.size());
                std::cerr << "    expected: \"" << expected_text.substr(ctx_start, ctx_end_e - ctx_start) << "\"" << std::endl;
                std::cerr << "    actual:   \"" << actual_text.substr(ctx_start, ctx_end_a - ctx_start) << "\"" << std::endl;
            }
            continue;
        }
        if (type != "basic") {
            result.skipped++;
            continue;
        }

        result.total++;
        std::string input = j.value("input", "");

        // Get expected ids_raw
        std::vector<int> expected_ids;
        if (j.contains("ids_raw") && j["ids_raw"].is_array()) {
            size_t n = j["ids_raw"].size();
            for (size_t i = 0; i < n; i++) {
                expected_ids.push_back(j["ids_raw"][i].get<int>());
            }
        }

        // Get expected decoded_full
        std::string expected_decoded = j.value("decoded_full", "");

        // Encode (without special tokens — encode() adds prefix_tokens automatically,
        // so we need to strip them for comparison with ids_raw)
        std::vector<int> actual_ids = tok->encode(input);

        // Strip prefix tokens if present
        // The prefix tokens are added by Tokenizer::encode() wrapper.
        // For models with BOS/prefix tokens, actual_ids will have them but ids_raw won't.
        // We detect and strip by checking if actual_ids starts with tokens not in ids_raw.
        // Simpler approach: compare with ids_full first, then ids_raw.
        bool encode_pass = false;

        if (actual_ids == expected_ids) {
            encode_pass = true;
        } else {
            // Try comparing with ids_full (which may include BOS/EOS)
            std::vector<int> expected_full;
            if (j.contains("ids_full") && j["ids_full"].is_array()) {
                size_t n = j["ids_full"].size();
                for (size_t i = 0; i < n; i++) {
                    expected_full.push_back(j["ids_full"][i].get<int>());
                }
            }
            if (!expected_full.empty() && actual_ids == expected_full) {
                encode_pass = true;
            }
        }

        // Decode: use full-sequence decode (uses decoder chain for proper spacing)
        std::string actual_decoded = tok->decode(expected_ids);
        // Full-sequence decode may have an extra leading space (from ▁/Metaspace)
        // that HuggingFace strips. Allow this difference.
        bool decode_pass = (actual_decoded == expected_decoded);
        if (!decode_pass && !actual_decoded.empty() && actual_decoded[0] == ' ' &&
            actual_decoded.substr(1) == expected_decoded) {
            decode_pass = true;
        }

        if (encode_pass && decode_pass) {
            result.passed++;
        } else {
            result.failed++;
            std::cerr << RED << "  FAIL line " << line_no << ": \"" << input << "\"" << RESET << std::endl;
            if (!encode_pass) {
                std::cerr << "    encode expected: " << ids_to_str(expected_ids) << std::endl;
                std::cerr << "    encode actual:   " << ids_to_str(actual_ids) << std::endl;
            }
            if (!decode_pass) {
                std::cerr << "    decode expected: \"" << expected_decoded << "\"" << std::endl;
                std::cerr << "    decode actual:   \"" << actual_decoded << "\"" << std::endl;
            }
        }
    }

    delete tok;
    return result;
}

int main(int argc, char* argv[]) {
    std::string models_path = "./models";
    std::string filter = "";

    if (argc > 1) models_path = argv[1];
    if (argc > 2) filter = argv[2];

    auto models = list_dirs(models_path);
    if (models.empty()) {
        std::cerr << "No model directories found in: " << models_path << std::endl;
        return 1;
    }

    std::vector<TestResult> results;
    int total_models = 0, total_passed = 0, total_failed = 0, total_skipped = 0;
    std::vector<std::string> failed_models;

    for (const auto& model : models) {
        if (!filter.empty() && model.find(filter) == std::string::npos) continue;

        std::cout << "Testing: " << model << " ..." << std::endl;
        auto result = test_model(models_path, model);
        results.push_back(result);

        if (result.total > 0) {
            total_models++;
            total_passed += result.passed;
            total_failed += result.failed;
            total_skipped += result.skipped;

            const char* color = (result.failed == 0) ? GREEN : RED;
            std::cout << color << "  " << result.passed << "/" << result.total
                      << " passed" << RESET;
            if (result.failed > 0) {
                std::cout << RED << ", " << result.failed << " failed" << RESET;
                failed_models.push_back(model);
            }
            if (result.skipped > 0) {
                std::cout << YELLOW << ", " << result.skipped << " skipped" << RESET;
            }
            std::cout << std::endl;
        }
    }

    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Summary: " << total_models << " models tested" << std::endl;
    std::cout << GREEN << "  Passed: " << total_passed << RESET << std::endl;
    if (total_failed > 0) {
        std::cout << RED << "  Failed: " << total_failed << RESET << std::endl;
    }
    if (total_skipped > 0) {
        std::cout << YELLOW << "  Skipped: " << total_skipped << RESET << std::endl;
    }
    if (!failed_models.empty()) {
        std::cout << RED << "\nFailed models:" << RESET << std::endl;
        for (const auto& m : failed_models) {
            std::cout << RED << "  - " << m << RESET << std::endl;
        }
    }
    std::cout << "========================================" << std::endl;

    return total_failed > 0 ? 1 : 0;
}
