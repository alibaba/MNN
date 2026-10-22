//
//  needlebench.cpp
//
//  Created by MNN on 2026/10/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "token_compressor.hpp"
#include "llm/llm.hpp"
#include "llmconfig.hpp"
#include "tokenizer/tokenizer.hpp"
#include "ujson.hpp"
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

using namespace MNN::Transformer;
using TokenCompressor = MNN::Transformer::TokenCompression::Compressor;
using Json = ujson::json;

static bool readFile(const std::string& path, std::string& text) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    std::ostringstream stream;
    stream << file.rdbuf();
    text = stream.str();
    return !file.bad();
}
static bool textField(const Json& row, const std::string& key, std::string& value) {
    if (row.contains(key) && row.contains(key + "_file")) {
        return false;
    }
    if (row.contains(key) && !row[key].is_string()) {
        return false;
    }
    if (row.contains(key + "_file") && !row[key + "_file"].is_string()) {
        return false;
    }
    if (row.contains(key + "_file")) {
        return readFile(row[key + "_file"].get<std::string>(), value);
    }
    value = row.value(key, "");
    return true;
}

// Match UTF-8 replacement at token-slice boundaries used by the dataset builder.
static std::string decodeBackground(Tokenizer* tokenizer, const std::vector<int>& ids) {
    const auto bytes = tokenizer->decode(ids);
    std::string text;
    text.reserve(bytes.size());
    for (size_t i = 0; i < bytes.size();) {
        const unsigned char first = bytes[i];
        if (first < 0x80) {
            text += bytes[i++];
            continue;
        }
        size_t width = first >= 0xC2 && first <= 0xDF   ? 2
                       : first >= 0xE0 && first <= 0xEF ? 3
                       : first >= 0xF0 && first <= 0xF4 ? 4
                                                        : 0;
        size_t used = 1;
        while (width && used < width && i + used < bytes.size()) {
            const unsigned char next = bytes[i + used];
            if (next < 0x80 || next > 0xBF ||
                (used == 1 && ((first == 0xE0 && next < 0xA0) || (first == 0xED && next > 0x9F) ||
                               (first == 0xF0 && next < 0x90) || (first == 0xF4 && next > 0x8F)))) {
                break;
            }
            ++used;
        }
        if (width && used == width) {
            text.append(bytes, i, used);
        } else {
            text += "\xEF\xBF\xBD";
        }
        i += used;
    }
    return text;
}
static bool trimCharacters(std::string& text, size_t count, bool head) {
    size_t begin = 0, end = text.size();
    while (count--) {
        if (begin == end) {
            return false;
        }
        if (head) {
            ++begin;
            while (begin < end && (static_cast<unsigned char>(text[begin]) & 0xC0) == 0x80) {
                ++begin;
            }
        } else {
            --end;
            while (end > begin && (static_cast<unsigned char>(text[end]) & 0xC0) == 0x80) {
                --end;
            }
        }
    }
    text = text.substr(begin, end - begin);
    return true;
}
static bool tokenCountField(const Json& object, const std::string& key, size_t& count, bool optional = false) {
    if (!object.contains(key)) {
        return optional;
    }
    const auto value = object[key];
    if (!value.is_number_integer()) {
        return false;
    }
    const auto number = value.get<int64_t>();
    if (number < 0 || number > std::numeric_limits<int>::max()) {
        return false;
    }
    count = static_cast<size_t>(number);
    return true;
}
class ContextBuilder {
public:
    bool read(const Json& row, Tokenizer* tokenizer, std::string& text) {
        const int modes = row.contains("context") + row.contains("context_file") + row.contains("context_builder");
        if (modes != 1) {
            return false;
        }
        if (!row.contains("context_builder")) {
            return textField(row, "context", text);
        }
        const auto spec = row["context_builder"];
        size_t count = 0, insertion = 0, head = 0, tail = 0;
        if (!spec.is_object() || !spec.contains("corpus_file") || !spec["corpus_file"].is_string() ||
            !spec.contains("insert_text") || !spec["insert_text"].is_string() ||
            !tokenCountField(spec, "background_tokens", count) || !tokenCountField(spec, "insert_at", insertion) ||
            !tokenCountField(spec, "trim_head_chars", head, true) ||
            !tokenCountField(spec, "trim_tail_chars", tail, true) || insertion > count) {
            return false;
        }
        const auto path = spec["corpus_file"].get<std::string>();
        if (!loadCorpus(path)) {
            return false;
        }
        const auto& documents = mCorpora.at(path);
        if (spec.contains("order") && (!spec["order"].is_array() || spec["order"].size() != documents.size())) {
            return false;
        }
        std::vector<bool> used(documents.size(), false);
        std::string corpus;
        for (size_t i = 0; i < documents.size(); ++i) {
            size_t index = i;
            if (spec.contains("order")) {
                const auto value = spec["order"][i];
                if (!value.is_number_integer() || value.get<int64_t>() < 0 ||
                    static_cast<uint64_t>(value.get<int64_t>()) >= documents.size()) {
                    return false;
                }
                index = static_cast<size_t>(value.get<int64_t>());
            }
            if (used[index]) {
                return false;
            }
            used[index] = true;
            if (i) {
                corpus += '\n';
            }
            corpus += documents[index];
        }
        const auto tokens = tokenizer->encode(corpus, false);
        if (tokens.empty()) {
            return false;
        }
        std::vector<int> background;
        background.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            background.push_back(tokens[i % tokens.size()]);
        }
        std::string before =
            decodeBackground(tokenizer, std::vector<int>(background.begin(), background.begin() + insertion));
        std::string after =
            decodeBackground(tokenizer, std::vector<int>(background.begin() + insertion, background.end()));
        if (!trimCharacters(before.empty() ? after : before, head, true) ||
            !trimCharacters(after.empty() ? before : after, tail, false)) {
            return false;
        }
        text = before + spec["insert_text"].get<std::string>() + after;
        return true;
    }
    Json order(const std::string& path, uint32_t seed) {
        if (!loadCorpus(path) || mCorpora.at(path).size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return Json::array();
        }
        std::vector<int> indices;
        for (size_t i = 0; i < mCorpora.at(path).size(); ++i) {
            indices.push_back(static_cast<int>(i));
        }
        std::mt19937 random(seed);
        // Explicit bounded sampling avoids std::shuffle differences across standard libraries.
        for (size_t n = indices.size(); n > 1; --n) {
            uint32_t bound = static_cast<uint32_t>(n), threshold = (uint32_t(0) - bound) % bound, value;
            do {
                value = random();
            } while (value < threshold);
            std::swap(indices[n - 1], indices[value % bound]);
        }
        Json result = Json::array();
        for (int index : indices) {
            result.push_back(index);
        }
        return result;
    }

private:
    bool loadCorpus(const std::string& path) {
        if (!mCorpora.count(path)) {
            std::ifstream corpus(path);
            if (!corpus) {
                return false;
            }
            std::vector<std::string> documents;
            std::string line;
            while (std::getline(corpus, line)) {
                if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
                    continue;
                }
                const auto document = Json::parse(line);
                if (!document.is_object() || !document.contains("text") || !document["text"].is_string()) {
                    return false;
                }
                documents.push_back(document["text"].get<std::string>());
            }
            if (corpus.bad() || documents.empty()) {
                return false;
            }
            mCorpora.emplace(path, std::move(documents));
        }
        return true;
    }
    std::map<std::string, std::vector<std::string>> mCorpora;
};

static std::string relativeTo(const std::string& base, const std::string& path) {
    if (path.empty() || path[0] == '/' || path[0] == '\\' || (path.size() > 1 && path[1] == ':')) {
        return path;
    }
    return base + path;
}

// Raw HF single-needle data and exact-replay JSONL share the same downstream case format.
class CaseReader {
public:
    bool open(const std::string& path) {
        mInput.open(path);
        if (!mInput) {
            return false;
        }
        while (std::getline(mInput, mPending) && mPending.find_first_not_of(" \t\r") == std::string::npos) {
        }
        if (mPending.find_first_not_of(" \t\r") == std::string::npos) {
            return false;
        }
        auto trimmed = mPending.substr(mPending.find_first_not_of(" \t\r"));
        bool multiline = trimmed.find_last_not_of(" \t\r") == 0 && trimmed[0] == '{';
        Json spec = multiline ? Json() : Json::parse(mPending);
        if (!multiline && (!spec.is_object() || spec.contains("case_id") || !spec.contains("data_dir"))) {
            return true;
        }
        if (multiline) {
            std::ostringstream rest;
            rest << mPending << '\n' << mInput.rdbuf();
            spec = Json::parse(rest.str());
        }
        if (!spec.is_object() || !spec.contains("data_dir") || !spec["data_dir"].is_string() ||
            spec["data_dir"].get<std::string>().empty() || !tokenCountField(spec, "length", mLength, true) ||
            !mLength || !tokenCountField(spec, "seed", mSeed, true)) {
            return false;
        }
        for (auto it = spec.begin(); it != spec.end(); ++it) {
            const auto key = it.key();
            if (key != "data_dir" && key != "length" && key != "depths" && key != "seed" && key != "language") {
                return false;
            }
        }
        std::string language;
        if (spec.contains("language")) {
            if (!spec["language"].is_string()) {
                return false;
            }
            language = spec["language"].get<std::string>();
            if (language == "en") {
                language = "English";
            }
            if (language == "zh") {
                language = "Chinese";
            }
            if (language != "English" && language != "Chinese") {
                return false;
            }
        }
        if (spec.contains("depths")) {
            auto depths = spec["depths"];
            if (!depths.is_array() || depths.empty()) {
                return false;
            }
            mDepths.clear();
            std::set<double> unique;
            for (const auto& value : depths) {
                if (!value.is_number()) {
                    return false;
                }
                double depth = value.get<double>();
                if (!std::isfinite(depth) || depth < 0 || depth > 100 || !unique.insert(depth).second) {
                    return false;
                }
                mDepths.push_back(depth);
            }
        }
        auto slash = path.find_last_of("/\\");
        mData = relativeTo(slash == std::string::npos ? "" : path.substr(0, slash + 1),
                           spec["data_dir"].get<std::string>());
        if (mData.back() != '/' && mData.back() != '\\') {
            mData += '/';
        }
        std::ifstream needles(mData + "needles.jsonl");
        if (!needles) {
            return false;
        }
        std::string line;
        size_t index = 0;
        while (std::getline(needles, line)) {
            if (line.find_first_not_of(" \t\r") == std::string::npos) {
                continue;
            }
            auto needle = Json::parse(line);
            for (const char* key : {"needle", "retrieval_question", "arg2", "language"}) {
                if (!needle.is_object() || !needle.contains(key) || !needle[key].is_string() ||
                    needle[key].get<std::string>().empty()) {
                    return false;
                }
            }
            const auto lang = needle["language"].get<std::string>();
            if (lang != "English" && lang != "Chinese") {
                return false;
            }
            if (needle.contains("gold_standard_answer") && !needle["gold_standard_answer"].is_string()) {
                return false;
            }
            if (language.empty() || language == lang) {
                mNeedles.push_back({index, needle});
            }
            ++index;
        }
        mRaw = true;
        return !needles.bad() && !mNeedles.empty();
    }

    bool next(ContextBuilder& builder, Json& row) {
        if (!mRaw) {
            std::string line;
            do {
                if (!mPending.empty()) {
                    line.swap(mPending);
                } else if (!std::getline(mInput, line)) {
                    return false;
                }
            } while (line.empty());
            row = Json::parse(line);
            return true;
        }
        if (mPosition == mNeedles.size()) {
            return false;
        }
        const auto& entry = mNeedles[mPosition];
        const auto& needle = entry.second;
        const auto language = needle["language"].get<std::string>();
        const auto corpus = mData + (language == "English" ? "PaulGrahamEssays.jsonl" : "zh_finance.jsonl");
        auto order = builder.order(corpus, static_cast<uint32_t>(mSeed + entry.first));
        if (order.empty()) {
            mFailed = true;
            return false;
        }
        double depth = mDepths[mDepth];
        std::ostringstream id;
        id.precision(17);
        id << (language == "English" ? "En-" : "Ch-") << entry.first << "-L" << mLength << "-D" << depth;
        row = Json::object();
        row["case_id"] = id.str();
        row["prefix"] = "<document>\n";
        row["suffix"] = "\n</document>\n\n" + needle["retrieval_question"].get<std::string>();
        row["needle"] = needle["needle"];
        row["arg2"] = needle["arg2"];
        row["hf_source"] = needle;
        Json profile = Json::object();
        profile["name"] = "hf-single-v1";
        profile["length_limit"] = static_cast<int64_t>(mLength);
        profile["depth_percent"] = depth;
        profile["seed"] = static_cast<int64_t>(mSeed);
        row["dataset"] = profile;
        Json construction = Json::object();
        construction["corpus_file"] = corpus;
        construction["order"] = order;
        construction["insert_text"] = "\n" + needle["needle"].get<std::string>() + "\n";
        row["context_builder"] = construction;
        if (++mDepth == mDepths.size()) {
            mDepth = 0;
            ++mPosition;
        }
        return true;
    }

    bool good() const { return !mFailed && !mInput.bad(); }

private:
    std::ifstream mInput;
    std::string mPending, mData;
    std::vector<std::pair<size_t, Json>> mNeedles;
    std::vector<double> mDepths{50};
    size_t mLength = 1000000, mSeed = 42, mPosition = 0, mDepth = 0;
    bool mRaw = false, mFailed = false;
};

static bool constructCase(Json& row, ContextBuilder& builder, Tokenizer* tokenizer, std::string& context,
                          std::string& prefix, std::string& suffix) {
    if (!textField(row, "prefix", prefix) || !textField(row, "suffix", suffix)) {
        return false;
    }
    if (!row.contains("dataset") || !row.contains("hf_source") || !row.contains("context_builder") ||
        row["context_builder"].contains("background_tokens")) {
        return builder.read(row, tokenizer, context);
    }
    const size_t target = static_cast<size_t>(row["dataset"]["length_limit"].get<int64_t>());
    const double depth = row["dataset"]["depth_percent"].get<double>();
    auto count = [&](const std::string& text) {
        return tokenizer->encode(tokenizer->apply_chat_template(ChatMessages{{"user", text}}, true), false).size();
    };
    const size_t fixed = count(prefix + row["context_builder"]["insert_text"].get<std::string>() + suffix);
    if (target <= fixed) {
        return false;
    }
    size_t budget = target - fixed, best = 0;
    Json bestSpec;
    std::string bestContext;
    std::set<size_t> attempted;
    for (int attempt = 0; attempt < 8 && attempted.insert(budget).second; ++attempt) {
        auto spec = Json::parse(row["context_builder"].dump());
        spec["background_tokens"] = static_cast<int64_t>(budget);
        spec["insert_at"] = static_cast<int64_t>(std::floor(budget * depth / 100.0));
        row["context_builder"] = spec;
        if (!builder.read(row, tokenizer, context)) {
            return false;
        }
        size_t actual = count(prefix + context + suffix);
        if (actual <= target && actual > best) {
            best = actual;
            bestSpec = spec;
            bestContext = context;
        }
        if (actual <= target && target - actual <= 8) {
            break;
        }
        int64_t next = static_cast<int64_t>(budget) + static_cast<int64_t>(target) - static_cast<int64_t>(actual);
        if (next < 0 || next > std::numeric_limits<int>::max()) {
            return false;
        }
        budget = static_cast<size_t>(next);
        while (budget && attempted.count(budget)) {
            --budget;
        }
    }
    if (!best || target - best > 8) {
        return false;
    }
    row["context_builder"] = bestSpec;
    row["dataset"]["raw_prompt_tokens"] = static_cast<int64_t>(best);
    context.swap(bestContext);
    return true;
}

static double seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
static int64_t peakRssBytes() {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return -1;
    }
#if defined(__APPLE__)
    return usage.ru_maxrss;
#else
    return usage.ru_maxrss * 1024LL;
#endif
#else
    return -1;
#endif
}

static void printUsage(std::ostream& output) {
    output << R"(Usage: needlebench config.json input.json[l] results.jsonl [options]

Options:
  --max-input N    Full input token limit, including template (default: 64000)
  --max-new N      Maximum generated tokens (default: 1024)
  --limit N        Run the first N generated cases (default: all)
  --auto-compress  Use the LLM auto-compression path (default: context-only)
  --compress-only  Check compression without loading model weights or inference
  --no-compress    Skip compression, still check the input limit; excludes --auto-compress
  --trace          Save token IDs and selection details
  -h, --help       Show this help

Input:
  HF Single-Needle dataset config, constructing long contexts at runtime:
    {"data_dir":"NeedleBench","length":1000000,"depths":[10,50,90],"seed":42}
  data_dir is relative to the config file and contains needles.jsonl plus
  PaulGrahamEssays.jsonl / zh_finance.jsonl. Optional language: English or Chinese.
  Defaults: length=1000000, depths=[50], seed=42.
  length includes the needle, question and chat template; depths are percentages in [0,100].

  JSONL replay: one case per line, with context, context_file, or context_builder.
  Replay file paths are relative to the working directory.

Examples:
  needlebench MODEL/config.json bench.json results.jsonl --auto-compress --max-input 64000
  needlebench MODEL/config.json bench.json check.jsonl --auto-compress --compress-only --limit 3

Use a new results.jsonl path. Backend, threads and prefill chunk come from config.json.
All numeric options require positive integers. --max-input overrides config.json's token_compress.
This tool uses hf-single-v1 construction and keyword matching, not the OpenCompass evaluation protocol.
keyword_hit is an exact keyword check, not the official NeedleBench score.
See docs/transformers/llm.md for input fields and token_compress limits.
)";
}

int main(int argc, const char** argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(std::cout);
        return 0;
    }
    if (argc < 4) {
        printUsage(std::cerr);
        return 2;
    }
    bool compressOnly = false, autoCompress = false, noCompress = false, trace = false;
    int maxNew = 1024, limit = 0;
    TokenCompressor::Options options;
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--compress-only") {
            compressOnly = true;
        } else if (arg == "--auto-compress") {
            autoCompress = true;
        } else if (arg == "--no-compress") {
            noCompress = true;
        } else if (arg == "--trace") {
            trace = true;
        } else if (i + 1 < argc && (arg == "--max-input" || arg == "--max-new" || arg == "--limit")) {
            char* end = nullptr;
            errno = 0;
            long value = std::strtol(argv[++i], &end, 10);
            if (!end || *end || errno == ERANGE || value <= 0 || value > std::numeric_limits<int>::max()) {
                std::cerr << "Invalid numeric argument\n";
                return 2;
            }
            if (arg == "--max-input") {
                options.maxTokens = value;
            } else if (arg == "--max-new") {
                maxNew = value;
            } else {
                limit = value;
            }
        } else {
            std::cerr << "Unknown/incomplete argument: " << arg << '\n';
            return 2;
        }
    }
    if (autoCompress && noCompress) {
        std::cerr << "--auto-compress and --no-compress are mutually exclusive\n";
        return 2;
    }
    CaseReader input;
    std::ifstream previousOutput(argv[3]);
    if (previousOutput.good() || !input.open(argv[2])) {
        std::cerr << "Invalid input/dataset config or output already exists; use a new result path\n";
        return 2;
    }
    LlmConfig config(argv[1]);
    std::unique_ptr<Tokenizer> tokenizer(Tokenizer::createTokenizer(config.tokenizer_file()));
    if (!tokenizer) {
        std::cerr << "Unable to load tokenizer\n";
        return 2;
    }
    if (config.config_.contains("jinja")) {
        auto jinja = config.config_["jinja"];
        auto context = jinja.contains("context") ? jinja["context"] : Json::object();
        context["enable_thinking"] = false;
        tokenizer->set_chat_template(jinja.value("chat_template", ""), jinja.value("eos", ""), context.dump());
    }
    if (tokenizer->chat_template().empty()) {
        std::cerr << "An explicit exported chat template is required\n";
        return 2;
    }
    auto encode = [&](const std::string& text) { return tokenizer->encode(text, false); };
    auto render = [&](const std::string& text) {
        return tokenizer->apply_chat_template(ChatMessages{{"user", text}}, true);
    };
    auto count = [&](const std::string& text) { return encode(render(text)).size(); };
    TokenCompressor compressor(TokenCompression::sentenceDelimiters(tokenizer.get()), encode("\n[…]\n"));
    std::unique_ptr<Llm> llm;
    if (!compressOnly) {
        llm.reset(Llm::createLLM(argv[1]));
        if (!llm->set_config("{\"sampler_type\":\"greedy\",\"temperature\":0,\"presence_penalty\":0,"
                             "\"repetition_penalty\":1,\"reuse_kv\":false,\"prompt_cache\":false,"
                             "\"token_compress\":0,\"kvcache_mmap\":false,\"async\":false}")) {
            return 2;
        }
        Json runtime = Json::object();
        runtime["token_compress"] = autoCompress ? static_cast<int64_t>(options.maxTokens) : 0;
        runtime["prompt_cache"] = false;
        auto jinja = config.config_.contains("jinja") ? config.config_["jinja"] : Json::object();
        auto context = jinja.contains("context") ? jinja["context"] : Json::object();
        context["enable_thinking"] = false;
        jinja["context"] = context;
        runtime["jinja"] = jinja;
        if (!llm->set_config(runtime.dump())) {
            return 2;
        }
        if (!llm->load()) {
            std::cerr << "Model load failed\n";
            return 1;
        }
    }
    std::ofstream output(argv[3]);
    if (!output) {
        return 2;
    }
    ContextBuilder contextBuilder;
    std::set<std::string> ids;
    Json row;
    int completed = 0, invalid = 0, correct = 0, scored = 0;
    while ((!limit || completed < limit) && input.next(contextBuilder, row)) {
        if (!row.is_object() || !row.contains("case_id") || !row["case_id"].is_string()) {
            std::cerr << "Malformed case row\n";
            return 2;
        }
        std::string id = row["case_id"].get<std::string>();
        if (!ids.insert(id).second) {
            std::cerr << "Duplicate case: " << id << '\n';
            return 2;
        }
        auto constructStart = std::chrono::steady_clock::now();
        std::string context, prefix, suffix;
        if (!constructCase(row, contextBuilder, tokenizer.get(), context, prefix, suffix)) {
            std::cerr << "Missing/ambiguous input fields: " << id << '\n';
            return 2;
        }
        Json result = Json::object();
        result["case_id"] = id;
        result["construct_seconds"] = seconds(constructStart);
        result["runtime_constructed"] = row.contains("context_builder");
        if (row.contains("hf_source")) {
            result["case"] = row;
        }
        result["model_config"] = argv[1];
        result["max_new_tokens"] = maxNew;
        result["compressor"] = noCompress ? "none" : "token-coverage-cost-aware-idf-v6";
        result["auto_compress"] = autoCompress;
        auto start = std::chrono::steady_clock::now();
        TokenCompressor::Result compressed;
        auto source = encode(context);
        std::string prompt, error;
        std::vector<int> runtimeTokens;
        size_t inputTokens = 0;
        std::vector<std::pair<size_t, size_t>> calibration;
        bool ok = false;
        if (autoCompress) {
            prompt = prefix + context + suffix;
            runtimeTokens =
                TokenCompression::encodePrompt(tokenizer.get(), render(prompt), static_cast<int>(options.maxTokens));
            ok = !runtimeTokens.empty() && runtimeTokens.size() <= options.maxTokens;
            inputTokens = runtimeTokens.size();
            compressed.tokens = runtimeTokens;
        } else if (noCompress || source.size() <= options.maxTokens) {
            prompt = prefix + context + suffix;
            inputTokens = count(prompt);
            ok = inputTokens <= options.maxTokens;
            if (ok) {
                compressed.tokens = source;
                if (!source.empty()) {
                    compressed.spans.push_back({0, source.size()});
                    compressed.selected.push_back(0);
                }
            }
        }
        if (!ok && !noCompress && !autoCompress) {
            size_t wrapper = count(prefix + suffix);
            if (wrapper < options.maxTokens) {
                size_t budget = options.maxTokens - wrapper;
                budget = budget > 16 ? budget - 16 : budget;
                std::set<size_t> attempted;
                for (int attempt = 0; attempt < 6 && budget && attempted.insert(budget).second; ++attempt) {
                    auto tokenOptions = options;
                    tokenOptions.maxTokens = budget;
                    TokenCompressor::Result candidate;
                    if (!compressor.compress(source, tokenOptions, candidate)) {
                        break;
                    }
                    std::string text = prefix + tokenizer->decode(candidate.tokens) + suffix;
                    size_t actual = count(text);
                    calibration.emplace_back(budget, actual);
                    if (actual <= options.maxTokens && (!ok || actual > inputTokens)) {
                        ok = true;
                        inputTokens = actual;
                        prompt = std::move(text);
                        compressed = std::move(candidate);
                    }
                    if (actual <= options.maxTokens && options.maxTokens - actual <= 128) {
                        break;
                    }
                    if (actual > options.maxTokens) {
                        size_t reduction = actual - options.maxTokens + 16;
                        budget = budget > reduction ? budget - reduction : 0;
                    } else {
                        budget += options.maxTokens - actual;
                    }
                }
            }
        }
        if (!ok) {
            error = "Input cap cannot be satisfied without dropping immutable wrappers or mandatory boundaries";
        }
        result["compress_seconds"] = seconds(start);
        result["input_tokens"] = static_cast<int64_t>(inputTokens);
        result["max_input_tokens"] = static_cast<int64_t>(options.maxTokens);
        result["source_tokens"] = static_cast<int64_t>(source.size());
        result["source_bytes"] = static_cast<int64_t>(context.size());
        result["prompt"] = prompt;
        result["rendered_prompt"] = ok ? (autoCompress ? tokenizer->decode(runtimeTokens) : render(prompt)) : "";
        result["reserve_units"] = static_cast<int64_t>(compressed.reserveUnits);
        if (trace) {
            result["token_spans"] = Json::array();
            for (auto p : compressed.spans) {
                result["token_spans"].push_back(Json{static_cast<int64_t>(p.first), static_cast<int64_t>(p.second)});
            }
            result["compressed_tokens"] = Json::array();
            for (int token : compressed.tokens) {
                result["compressed_tokens"].push_back(token);
            }
            result["selected"] = Json::array();
            for (auto i : compressed.selected) {
                result["selected"].push_back(static_cast<int64_t>(i));
            }
            result["calibration"] = Json::array();
            for (auto p : calibration) {
                result["calibration"].push_back(Json{static_cast<int64_t>(p.first), static_cast<int64_t>(p.second)});
            }
        }
        std::string answer;
        bool valid = ok;
        if (!ok) {
            result["error"] = error;
        } else if (!compressOnly) {
            auto tokens = autoCompress ? std::vector<int>() : encode(render(prompt));
            llm->reset();
            start = std::chrono::steady_clock::now();
            std::ostringstream stream;
            if (autoCompress) {
                llm->response(ChatMessages{{"user", prompt}}, &stream, nullptr, maxNew);
            } else {
                llm->response(tokens, &stream, nullptr, maxNew);
            }
            const auto* ctx = llm->getContext();
            answer = ctx->generate_str;
            result["output"] = answer;
            result["inference_seconds"] = seconds(start);
            result["runtime_prompt_tokens"] = ctx->prompt_len;
            result["generated_tokens"] = static_cast<int64_t>(ctx->output_tokens.size());
            result["status"] = static_cast<int>(ctx->status);
            result["prefill_us"] = ctx->prefill_us;
            result["decode_us"] = ctx->decode_us;
            valid = ctx->status == LlmStatus::NORMAL_FINISHED && ctx->prompt_len == static_cast<int>(inputTokens) &&
                    !answer.empty();
            if (!valid) {
                result["error"] = "Runtime error, token mismatch, empty answer, or output-budget exhaustion";
            }
        }
        // Gold is consumed only for auditing/scoring, after compression and generation.
        if (row.contains("needle")) {
            result["needle_retained"] = result["rendered_prompt"].get<std::string>().find(
                                            row["needle"].get<std::string>()) != std::string::npos;
        }
        if (row.contains("arg2") && !compressOnly) {
            std::string finalAnswer = answer;
            auto thinkEnd = finalAnswer.rfind("</think>");
            if (thinkEnd != std::string::npos) {
                finalAnswer = finalAnswer.substr(thinkEnd + 8);
            }
            bool hit = !row["arg2"].get<std::string>().empty() &&
                       finalAnswer.find(row["arg2"].get<std::string>()) != std::string::npos;
            result["final_answer"] = finalAnswer;
            result["keyword_hit"] = hit;
            if (valid) {
                ++scored;
                if (hit) {
                    ++correct;
                }
            }
        }
        result["valid"] = valid;
        result["peak_rss_bytes"] = peakRssBytes();
        if (!valid) {
            ++invalid;
        }
        output << result.dump() << '\n';
        output.flush();
        if (!output) {
            std::cerr << "Result write failed\n";
            return 2;
        }
        ++completed;
        std::cerr << "case=" << id << " completed=" << completed << " valid=" << valid << " scored=" << scored
                  << " correct=" << correct << " invalid=" << invalid << '\n';
        if (!valid) {
            break;
        }
    }
    if (!input.good()) {
        std::cerr << "Input read failed\n";
        return 2;
    }
    std::cerr << "completed=" << completed << " scored=" << scored << " correct=" << correct << " invalid=" << invalid
              << '\n';
    return invalid || !completed ? 1 : 0;
}
