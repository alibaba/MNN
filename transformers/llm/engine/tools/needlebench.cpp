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
#include <iomanip>
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
            !tokenCountField(spec, "trim_tail_chars", tail, true) || insertion > count ||
            (spec.contains("corpus") && !spec["corpus"].is_string())) {
            return false;
        }
        const auto path = spec["corpus_file"].get<std::string>();
        const auto group = spec.value("corpus", "");
        if (!loadCorpus(path, group)) {
            return false;
        }
        const auto& documents = mCorpora.at({path, group});
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
    Json order(const std::string& path, uint32_t seed, const std::string& group = "") {
        if (!loadCorpus(path, group) ||
            mCorpora.at({path, group}).size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return Json::array();
        }
        std::vector<int> indices;
        for (size_t i = 0; i < mCorpora.at({path, group}).size(); ++i) {
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
    bool loadCorpus(const std::string& path, const std::string& group) {
        const auto key = std::make_pair(path, group);
        if (!mCorpora.count(key)) {
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
                if (!document.is_object() || !document.contains("text") || !document["text"].is_string() ||
                    (document.contains("corpus") && !document["corpus"].is_string())) {
                    return false;
                }
                if (group.empty() || !document.contains("corpus") || document["corpus"].get<std::string>() == group) {
                    documents.push_back(document["text"].get<std::string>());
                }
            }
            if (corpus.bad() || documents.empty()) {
                return false;
            }
            mCorpora.emplace(key, std::move(documents));
        }
        return true;
    }
    std::map<std::pair<std::string, std::string>, std::vector<std::string>> mCorpora;
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
    bool open(const std::string& path, const std::string& corpus) {
        mBase = base_dir(path);
        mCorpus = corpus;
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
        if (!multiline && spec.is_object() && !spec.contains("case_id") && spec.contains("needle") &&
            spec.contains("retrieval_question") && !mCorpus.empty()) {
            mInput.clear();
            mInput.seekg(0);
            mRaw = true;
            return readNeedles(mInput, "");
        }
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
        mData = relativeTo(mBase, spec["data_dir"].get<std::string>());
        if (mData.back() != '/' && mData.back() != '\\') {
            mData += '/';
        }
        std::ifstream needles(mData + "needles.jsonl");
        if (!needles) {
            return false;
        }
        mRaw = true;
        return readNeedles(needles, language);
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
            if (row.is_object()) {
                for (const char* key : {"context_file", "prefix_file", "suffix_file"}) {
                    if (row.contains(key) && row[key].is_string()) {
                        row[key] = relativeTo(mBase, row[key].get<std::string>());
                    }
                }
                if (row.contains("context_builder") && row["context_builder"].is_object()) {
                    auto construction = row["context_builder"];
                    if (!mCorpus.empty()) {
                        construction["corpus_file"] = mCorpus;
                    } else if (construction.contains("corpus_file") && construction["corpus_file"].is_string()) {
                        construction["corpus_file"] = relativeTo(mBase, construction["corpus_file"].get<std::string>());
                    }
                    row["context_builder"] = construction;
                }
            }
            return true;
        }
        if (mPosition == mNeedles.size()) {
            return false;
        }
        const auto& entry = mNeedles[mPosition];
        const auto& needle = entry.second;
        const auto language = needle["language"].get<std::string>();
        const auto corpus = mCorpus.empty()
                                ? mData + (language == "English" ? "PaulGrahamEssays.jsonl" : "zh_finance.jsonl")
                                : mCorpus;
        const auto group = mCorpus.empty() ? "" : language;
        auto order = builder.order(corpus, static_cast<uint32_t>(mSeed + entry.first), group);
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
        if (!group.empty()) {
            construction["corpus"] = group;
        }
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
    bool readNeedles(std::istream& needles, const std::string& language) {
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
        return !needles.bad() && !mNeedles.empty();
    }
    std::ifstream mInput;
    std::string mPending, mData, mBase, mCorpus;
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
    output << R"(Usage:
  needlebench -m config.json -c cases.jsonl -o results.jsonl [options]

Run one case with auto-compression to at most 64K tokens:
  needlebench -m MODEL/config.json -c cases.jsonl -b corpus.jsonl -o results.jsonl -ac -p 64000 -l 1

Input modes:
  Runtime construction:
    cases.jsonl contains the needle, question and construction parameters.
    corpus.jsonl contains filler rows: {"text":"background text"}.
    --background is required for raw HF rows or when context_builder.corpus_file is omitted.
    Optional filler labels: {"corpus":"English","text":"..."} / {"corpus":"Chinese","text":"..."}.
    Replay rows select a label with context_builder.corpus; raw HF rows use language.
    Raw HF Single-Needle defaults: length=1000000, depth=50%, seed=42.

  Existing long text:
    Each JSONL row contains context or context_file; --background is not needed.

  Dataset config / configured corpus path (also supported):
    JSONL context_builder.corpus_file supplies the filler path without --background.
    Or use bench.json: {"data_dir":"NeedleBench","length":1000000,"depths":[10,50,90],"seed":42}.
    data_dir contains needles.jsonl and PaulGrahamEssays.jsonl / zh_finance.jsonl.
    Optional language: English or Chinese. Defaults: length=1000000, depths=[50], seed=42.
    length includes the needle, question and template; depths are percentages in [0,100].

Paths inside case rows / bench.json are relative to that input file.
CLI paths, including --background, are relative to the working directory. No fixed data directory is required.

Options:
  -m,  --model FILE       Model config.json (required)
  -c,  --cases FILE       Case JSONL or dataset config JSON (required)
  -o,  --output FILE      New results JSONL path (required)
  -b,  --background FILE  Background JSONL; overrides context_builder.corpus_file
  -p,  --max-input N      Full input token limit, including template (default: 64000)
  -n,  --max-new N        Maximum generated tokens (default: 1024)
  -l,  --limit N          Run the first N generated cases (default: all)
  -ac, --auto-compress    Use the LLM auto-compression path (default: context-only)
  -co, --compress-only    Check compression without loading model weights or inference
  -nc, --no-compress      Skip compression, still check the input limit; excludes --auto-compress
  -tr, --trace            Save token IDs and selection details
  -h,  --help             Show this help

More examples:
  needlebench -m MODEL/config.json -c cases.jsonl -b corpus.jsonl -o check.jsonl -ac -co -l 3
  needlebench -m MODEL/config.json -c prepared.jsonl -o results.jsonl -ac -l 1
  needlebench --model MODEL/config.json --cases bench.json --output results.jsonl --auto-compress

All paths are named options; positional arguments are not supported.
Use a new results.jsonl path. Backend, threads and prefill chunk come from config.json.
All numeric options require positive integers. --max-input overrides config.json's token_compress.
Per-case timings and final totals are printed in seconds; prefill/decode are part of inference.
This tool uses hf-single-v1 construction and keyword matching, not the OpenCompass evaluation protocol.
keyword_hit is an exact keyword check, not the official NeedleBench score.
See docs/transformers/llm.md for input fields and token_compress limits.
)";
}

int main(int argc, const char** argv) {
    const auto runStart = std::chrono::steady_clock::now();
    bool compressOnly = false, autoCompress = false, noCompress = false, trace = false;
    std::string modelPath, casePath, outputPath, corpus;
    int maxNew = 1024, limit = 0;
    TokenCompressor::Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(std::cout);
            return 0;
        } else if (arg == "-m" || arg == "--model" || arg == "-c" || arg == "--cases" ||
                   arg == "-o" || arg == "--output" || arg == "-b" || arg == "--background") {
            if (i + 1 == argc || !argv[i + 1][0] || argv[i + 1][0] == '-') {
                std::cerr << "Missing/invalid path for " << arg << '\n';
                return 2;
            }
            std::string value = argv[++i];
            if (arg == "-m" || arg == "--model") {
                modelPath = value;
            } else if (arg == "-c" || arg == "--cases") {
                casePath = value;
            } else if (arg == "-o" || arg == "--output") {
                outputPath = value;
            } else {
                corpus = value;
            }
        } else if (arg == "-co" || arg == "--compress-only") {
            compressOnly = true;
        } else if (arg == "-ac" || arg == "--auto-compress") {
            autoCompress = true;
        } else if (arg == "-nc" || arg == "--no-compress") {
            noCompress = true;
        } else if (arg == "-tr" || arg == "--trace") {
            trace = true;
        } else if (i + 1 < argc && (arg == "-p" || arg == "--max-input" || arg == "-n" ||
                                   arg == "--max-new" || arg == "-l" || arg == "--limit")) {
            char* end = nullptr;
            errno = 0;
            long value = std::strtol(argv[++i], &end, 10);
            if (!end || *end || errno == ERANGE || value <= 0 || value > std::numeric_limits<int>::max()) {
                std::cerr << "Invalid numeric argument\n";
                return 2;
            }
            if (arg == "-p" || arg == "--max-input") {
                options.maxTokens = value;
            } else if (arg == "-n" || arg == "--max-new") {
                maxNew = value;
            } else {
                limit = value;
            }
        } else {
            std::cerr << "Unknown/incomplete argument: " << arg << '\n';
            return 2;
        }
    }
    if (modelPath.empty() || casePath.empty() || outputPath.empty()) {
        std::cerr << "--model, --cases and --output are required\n";
        printUsage(std::cerr);
        return 2;
    }
    if (autoCompress && noCompress) {
        std::cerr << "--auto-compress and --no-compress are mutually exclusive\n";
        return 2;
    }
    CaseReader input;
    std::ifstream previousOutput(outputPath);
    if (previousOutput.good() || !input.open(casePath, corpus)) {
        std::cerr << "Invalid input/dataset config or output already exists; use a new result path\n";
        return 2;
    }
    LlmConfig config(modelPath);
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
    double loadSeconds = 0;
    if (!compressOnly) {
        const auto loadStart = std::chrono::steady_clock::now();
        llm.reset(Llm::createLLM(modelPath));
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
        loadSeconds = seconds(loadStart);
    }
    std::ofstream output(outputPath);
    if (!output) {
        return 2;
    }
    ContextBuilder contextBuilder;
    std::set<std::string> ids;
    Json row;
    int completed = 0, invalid = 0, correct = 0, scored = 0;
    double constructTotal = 0, compressTotal = 0, inferenceTotal = 0;
    int64_t prefillTotal = 0, decodeTotal = 0;
    while (!limit || completed < limit) {
        const auto constructStart = std::chrono::steady_clock::now();
        if (!input.next(contextBuilder, row)) {
            break;
        }
        if (!row.is_object() || !row.contains("case_id") || !row["case_id"].is_string()) {
            std::cerr << "Malformed case row\n";
            return 2;
        }
        std::string id = row["case_id"].get<std::string>();
        if (!ids.insert(id).second) {
            std::cerr << "Duplicate case: " << id << '\n';
            return 2;
        }
        std::string context, prefix, suffix;
        if (!constructCase(row, contextBuilder, tokenizer.get(), context, prefix, suffix)) {
            std::cerr << "Missing/ambiguous input fields: " << id << '\n';
            return 2;
        }
        Json result = Json::object();
        result["case_id"] = id;
        const double constructSeconds = seconds(constructStart);
        result["construct_seconds"] = constructSeconds;
        result["runtime_constructed"] = row.contains("context_builder");
        if (row.contains("hf_source")) {
            result["case"] = row;
        }
        result["model_config"] = modelPath;
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
        const double compressSeconds = seconds(start);
        result["compress_seconds"] = compressSeconds;
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
        double inferenceSeconds = 0;
        if (!ok) {
            result["error"] = error;
        } else if (!compressOnly) {
            llm->reset();
            start = std::chrono::steady_clock::now();
            std::ostringstream stream;
            auto tokens = llm->tokenizer_encode(llm->apply_chat_template(ChatMessages{{"user", prompt}}));
            valid = !tokens.empty() && tokens.size() == inputTokens && tokens.size() <= options.maxTokens &&
                    (!autoCompress || tokens == runtimeTokens);
            result["runtime_prompt_tokens"] = static_cast<int64_t>(tokens.size());
            if (!valid) {
                result["error"] = "Runtime tokenization mismatch or input cap exceeded; inference skipped";
            } else {
                llm->response(tokens, &stream, nullptr, maxNew);
                const auto* ctx = llm->getContext();
                answer = ctx->generate_str;
                result["output"] = answer;
                result["runtime_prompt_tokens"] = ctx->prompt_len;
                result["generated_tokens"] = static_cast<int64_t>(ctx->output_tokens.size());
                result["status"] = static_cast<int>(ctx->status);
                result["prefill_us"] = ctx->prefill_us;
                result["decode_us"] = ctx->decode_us;
                prefillTotal += ctx->prefill_us;
                decodeTotal += ctx->decode_us;
                valid = ctx->status == LlmStatus::NORMAL_FINISHED && ctx->prompt_len == static_cast<int>(inputTokens) &&
                        !answer.empty();
                if (!valid) {
                    result["error"] = "Runtime error, token mismatch, empty answer, or output-budget exhaustion";
                }
            }
            inferenceSeconds = seconds(start);
            result["inference_seconds"] = inferenceSeconds;
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
        constructTotal += constructSeconds;
        compressTotal += compressSeconds;
        inferenceTotal += inferenceSeconds;
        std::cerr << "case=" << id << " completed=" << completed << " valid=" << valid << " scored=" << scored
                  << " correct=" << correct << " invalid=" << invalid << '\n'
                  << std::fixed << std::setprecision(3) << "time(s): construct=" << constructSeconds
                  << " compress=" << compressSeconds << " inference=" << inferenceSeconds
                  << " total=" << seconds(constructStart) << '\n';
        if (!valid) {
            break;
        }
    }
    if (!input.good()) {
        std::cerr << "Input read failed\n";
        return 2;
    }
    std::cerr << "completed=" << completed << " scored=" << scored << " correct=" << correct << " invalid=" << invalid
              << '\n'
              << std::fixed << std::setprecision(3) << "time(s): total=" << seconds(runStart)
              << " load=" << loadSeconds << " construct=" << constructTotal << " compress=" << compressTotal
              << " inference=" << inferenceTotal << " prefill=" << prefillTotal / 1000000.0
              << " decode=" << decodeTotal / 1000000.0 << '\n';
    return invalid || !completed ? 1 : 0;
}
