#include "llm/llm.hpp"
#include "core/MNNFileUtils.h"
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <fstream>
#include <sstream>
#include <regex>
#include <stdlib.h>
#include <initializer_list>
#include <rapidjson/document.h>
#include <thread>
#include <algorithm>
#include <numeric>


#define MNN_OPEN_TIME_TRACE


using namespace MNN::Transformer;

struct RuntimeParameters {
    std::vector<std::string>         model;
    std::vector<int>                 backends;
    std::vector<int>                 threads;
    std::vector<bool>                useMmap;
    std::vector<int>                 power;
    std::vector<int>                 precision;
    std::vector<int>                 memory;
    std::vector<int>                 dynamicOption;
};

struct TestParameters {
    std::vector<int>                 nPrompt;
    std::vector<int>                 nGenerate;
    std::vector<std::pair<int, int>> nPrompGen;
    std::vector<int>                 nRepeat;
    std::vector<std::string>         kvCache;
};

struct CommandParameters {
    std::string         model;
    int                 backend;
    int                 threads;
    bool                useMmap;
    int                 power;
    int                 precision;
    int                 memory;
    int                 dynamicOption;

    int                 nPrompt;
    int                 nGenerate;
    std::pair<int, int> nPrompGen;
    int                 nRepeat;
    std::string         kvCache;

};


static const RuntimeParameters runtimeParamsDefaults = {
    /* model                */ { "./Qwen2.5-1.5B-Instruct" },
    /* backends             */ { 0 },
    /* threads            */ { 4 },
    /* useMmap             */ { false },
    /* power                */ { 0 },
    /* precision            */ { 2 },
    /* memory               */ { 2 },
    /* dynamicOption       */ { 0 }
};


static const TestParameters testParamsDefaults = {
    /* nPrompt             */ { 512 },
    /* nGenerate                */ { 128 },
    /* nPrompGen                 */ {std::make_pair(512, 128)},
    /* nRepeat             */ { 5 },
    /* kvCache             */ { "false" }
};


struct commandParametersInstance {

    CommandParameters mCmdParam;

    commandParametersInstance(CommandParameters cmdParam) {
        mCmdParam.model          = cmdParam.model;
        mCmdParam.backend        = cmdParam.backend;
        mCmdParam.threads      = cmdParam.threads;
        mCmdParam.useMmap       = cmdParam.useMmap;
        mCmdParam.power          = cmdParam.power;
        mCmdParam.precision      = cmdParam.precision;
        mCmdParam.memory         = cmdParam.memory;
        mCmdParam.dynamicOption = cmdParam.dynamicOption;

        mCmdParam.nPrompt       = cmdParam.nPrompt;
        mCmdParam.nGenerate          = cmdParam.nGenerate;
        mCmdParam.nPrompGen           = cmdParam.nPrompGen;
        mCmdParam.nRepeat       = cmdParam.nRepeat;
        mCmdParam.kvCache       = cmdParam.kvCache;
    }

    CommandParameters get_cmd_parameters() const {
        return mCmdParam;
    }

    bool equal_runtime_params(const commandParametersInstance & other) const {
        return mCmdParam.model == other.mCmdParam.model &&
        mCmdParam.useMmap == other.mCmdParam.useMmap &&
        mCmdParam.power == other.mCmdParam.power &&
        mCmdParam.precision == other.mCmdParam.precision &&
        mCmdParam.memory == other.mCmdParam.memory &&
        mCmdParam.dynamicOption == other.mCmdParam.dynamicOption;
    }
};

template <typename T> static T avg(const std::vector<T> & v) {
    if (v.empty()) {
        return 0;
    }
    T sum = std::accumulate(v.begin(), v.end(), T(0));
    return sum / (T) v.size();
}

template <typename T> static T stdev(const std::vector<T> & v) {
    if (v.size() <= 1) {
        return 0;
    }
    T mean   = avg(v);
    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), T(0));
    T stdev  = std::sqrt(sq_sum / (T) (v.size() - 1) - mean * mean * (T) v.size() / (T) (v.size() - 1));
    return stdev;
}

template <class T> static std::string join(const std::vector<T> & values, const std::string & delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) {
            str << delim;
        }
    }
    return str.str();
}

struct TestInstance {
//    static const std::string build_commit;
    std::string              modelConfigFile;
    std::string              modelType;
    uint64_t                 modelSize;
    int                      threads;
    bool                     useMmap;
    int                      nPrompt;
    int                      nGenerate;
    std::vector<int64_t>     prefillUs;
    std::vector<int64_t>     decodeUs;
    std::vector<int64_t>     samplesUs;
    int                      backend;
    int                      precision;
    int                      power;
    int                      memory;

    TestInstance(const commandParametersInstance & instance) {

        modelConfigFile = instance.mCmdParam.model;
        threads         = instance.mCmdParam.threads;
        useMmap          = instance.mCmdParam.useMmap;
        nPrompt          = instance.mCmdParam.nPrompt;
        nGenerate             = instance.mCmdParam.nGenerate;
        backend           = instance.mCmdParam.backend;
        precision         = instance.mCmdParam.precision;
        memory            = instance.mCmdParam.memory;
        power             = instance.mCmdParam.power;
    }

    std::vector<double> getTokensPerSecond(int n_tokens, std::vector<int64_t> cost_us) const {
        std::vector<double> ts;
        std::transform(cost_us.begin(), cost_us.end(), std::back_inserter(ts), [n_tokens](int64_t t) { return 1e6 * n_tokens / t; });
        return ts;
    }

    double getAvgUs(std::vector<double> v) const { return ::avg(v); }
    double getStdevUs(std::vector<double> v) const { return ::stdev(v); }
    enum fieldType { STRING, BOOL, INT, FLOAT };

    static fieldType getFieldType(const std::string & field) {
        if (field == "threads") {
            return INT;
        }
        if (field == "useMmap") {
            return BOOL;
        }
        if (field == "t/s" || field == "modelSize" || field == "prefill&decode speed (tok/s)") {
            return FLOAT;
        }
        return STRING;
    }
};

static std::string pairString(const std::pair<int, int> & p) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "%d,%d", p.first, p.second);
    return buf;
}

template <typename T, typename F> static std::vector<std::string> transform2String(const std::vector<T> & values, F f) {
    std::vector<std::string> str_values;
    std::transform(values.begin(), values.end(), std::back_inserter(str_values), f);
    return str_values;
}

template<class T>
static std::vector<T> splitString(const std::string & str, char delim) {
    std::vector<T> values;
    std::istringstream str_stream(str);
    std::string token;
    while (std::getline(str_stream, token, delim)) {
        T value;
        std::istringstream tokenStream(token);
        tokenStream >> value;
        values.push_back(value);
    }
    return values;
}

struct Printer {
    virtual ~Printer() {}

    FILE * fout;

    virtual void printHeader(const RuntimeParameters & rp, const TestParameters & tp) { (void) rp; (void) tp; }

    virtual void printPerformance(const TestInstance & t) = 0;

//    virtual void print_footer() {}
};

struct markdownPrinter : public Printer {
    std::vector<std::string> fields;

    static int getFieldWidth(const std::string & field) {
        if (field == "model") {
            return -30;
        }
        if (field == "prefill&decode speed (tok/s)") {
            return 20;
        }
        if (field == "threads") {
            return 5;
        }
        if (field == "useMmap") {
            return 4;
        }
        if (field == "test") {
            return -13;
        }

        int width = std::max((int) field.length(), 10);

        if (TestInstance::getFieldType(field) == TestInstance::STRING) {
            return -width;
        }
        return width;
    }

    static std::string getFieldDisplayName(const std::string & field) {
        if (field == "useMmap") {
            return "mmap";
        }
        return field;
    }

    void printHeader(const RuntimeParameters & rp, const TestParameters & tp) override {
        // select fields to print
        fields.emplace_back("model");
        fields.emplace_back("modelSize");
        fields.emplace_back("backend");
        fields.emplace_back("threads");

        if (rp.precision.size() > 1) {
            fields.emplace_back("precision");
        }
        if (rp.memory.size() > 1) {
            fields.emplace_back("memory");
        }
        if (rp.dynamicOption.size() > 1) {
            fields.emplace_back("dynamicOption");
        }
        
        if (rp.useMmap.size() > 1 || rp.useMmap != runtimeParamsDefaults.useMmap) {
            fields.emplace_back("useMmap");
        }
        if (tp.kvCache[0] == "false") {
            fields.emplace_back("test");
            fields.emplace_back("t/s");
        } else {
            fields.emplace_back("llm_demo");
            fields.emplace_back("speed(tok/s)");
        }

        fprintf(fout, "|");
        for (const auto & field : fields) {
            fprintf(fout, " %*s |", getFieldWidth(field), getFieldDisplayName(field).c_str());
        }
        fprintf(fout, "\n");
        fprintf(fout, "|");
        for (const auto & field : fields) {
            int width = getFieldWidth(field);
            fprintf(fout, " %s%s |", std::string(std::abs(width) - 1, '-').c_str(), width > 0 ? ":" : "-");
        }
        fprintf(fout, "\n");
    }

    void printPerformance(const TestInstance & t) override {
        fprintf(fout, "|");
        for (const auto & field : fields) {
            std::string value;
            char        buf[128];
            if (field == "model") {
                value = t.modelType;
            } else if (field == "modelSize") {
                if (t.modelSize < 1024 * 1024 * 1024) {
                    snprintf(buf, sizeof(buf), "%.2f MiB", t.modelSize / 1024.0 / 1024.0);
                } else {
                    snprintf(buf, sizeof(buf), "%.2f GiB", t.modelSize / 1024.0 / 1024.0 / 1024.0);
                }
                value = buf;
            }  else if (field == "backend") {
                if (t.backend == 1) value = "METAL";
                else if (t.backend == 3) value = "OPENCL";
                else value = "CPU";
            } else if (field == "test") {
                if (t.nPrompt > 0 && t.nGenerate == 0) {
                    snprintf(buf, sizeof(buf), "pp%d", t.nPrompt);
                } else if (t.nGenerate > 0 && t.nPrompt == 0) {
                    snprintf(buf, sizeof(buf), "tg%d", t.nGenerate);
                } else {
                    snprintf(buf, sizeof(buf), "pp%d+tg%d", t.nPrompt, t.nGenerate);
                }
                value = buf;
            } else if (field == "llm_demo") {
                snprintf(buf, sizeof(buf), "prompt=%d<br>decode=%d", t.nPrompt, t.nGenerate);
                value = buf;
            } else if (field == "t/s") {
                auto spd = t.getTokensPerSecond(t.nPrompt + t.nGenerate, t.samplesUs);
                snprintf(buf, sizeof(buf), "%.2f ± %.2f", t.getAvgUs(spd), t.getStdevUs(spd));
                value = buf;
            } else if (field == "speed(tok/s)") {
                auto decode_speed = t.getTokensPerSecond(t.nGenerate, t.decodeUs);
                auto prefill_speed = t.getTokensPerSecond(t.nPrompt, t.prefillUs);
                snprintf(buf, sizeof(buf), "%.2f ± %.2f<br>%.2f ± %.2f", t.getAvgUs(prefill_speed), t.getStdevUs(prefill_speed), t.getAvgUs(decode_speed), t.getStdevUs(decode_speed));
                value = buf;
            } else if (field == "precision") {
                if (t.precision == 2) value = "Low";
                else if (t.precision == 0) value = "Normal";
                else value = "High";
            } else if (field == "memory") {
                if (t.memory == 2) value = "Low";
                else if (t.memory == 0) value = "Normal";
                else value = "High";
            } else if (field == "power") {
                if (t.power == 2) value = "Low";
                else if (t.power == 0) value = "Normal";
                else value = "High";
            } else if (field == "threads") {
                snprintf(buf, sizeof(buf), "%d", t.threads);
                value = buf;
            } else {
                assert(false);
                MNN_ERROR("llm bench print fields error\n");
                return;
            }

            int width = getFieldWidth(field);
            if (field == "prefill&decode speed (tok/s)" || field == "t/s") {
                // HACK: the utf-8 character is 2 bytes
                width += 1;
            }
            fprintf(fout, " %*s |", width, value.c_str());
        }
        fprintf(fout, "\n");
    }

//    void print_footer() override {
//        fprintf(fout, "\nbuild: %s \n", test::build_commit.c_str());
//    }
};

static FILE* openFile(const char* file, bool read) {
#if defined(_MSC_VER)
    wchar_t wFilename[1024];
    if (0 == MultiByteToWideChar(CP_ACP, 0, file, -1, wFilename, sizeof(wFilename))) {
        return nullptr;
    }
#if _MSC_VER >= 1400
    FILE* mFile = nullptr;
    if (read) {
        if (0 != _wfopen_s(&mFile, wFilename, L"r")) {
            return nullptr;
        }
    } else {
        if (0 != _wfopen_s(&mFile, wFilename, L"a")) {
            return nullptr;
        }
    }
    return mFile;
#else
    if (read) {
        return _wfopen(wFilename, L"r");
    } else {
        return _wfopen(wFilename, L"a");
    }
#endif
#else
    if (read) {
        return fopen(file, "r");
    } else {
        return fopen(file, "a");
    }
#endif
    return nullptr;
}

static std::vector<commandParametersInstance> get_cmd_params_instances(const RuntimeParameters & rp, const TestParameters& tp) {
    std::vector<commandParametersInstance> instances;

    // this ordering minimizes the number of times that each model needs to be reloaded
    // clang-format off
    for (const auto & m : rp.model)
    for (const auto & backend : rp.backends)
    for (const auto & precision : rp.precision)
    for (const auto & memory : rp.memory)
    for (const auto & power : rp.power)
    for (const auto & mmp : rp.useMmap)
    for (const auto & nt : rp.threads)
    for (const auto & dyop : rp.dynamicOption)
        if (tp.kvCache[0] == "true") { // MNN llm_demo test standard
            for (const auto & nPrompt : tp.nPrompt) {
                if (nPrompt == 0) {
                    continue;
                }
                for (const auto & nGenerate: tp.nGenerate) {
                    if (nGenerate == 0) {
                        continue;
                    }
                    CommandParameters tmpParam;
                    tmpParam.model = m;
                    tmpParam.backend = backend;
                    tmpParam.threads = nt;
                    tmpParam.power = power;
                    tmpParam.precision = precision;
                    tmpParam.memory = memory;
                    tmpParam.nPrompt = nPrompt;
                    tmpParam.nGenerate = nGenerate;
                    tmpParam.useMmap = mmp;
                    tmpParam.dynamicOption = dyop;
                    tmpParam.nRepeat = tp.nRepeat[0];
                    tmpParam.kvCache = "true";
                    auto instance = commandParametersInstance(tmpParam);
                    instances.push_back(instance);
                }
            }
        } else { // llama.cpp llama-bench's test standard
            for (const auto & nPrompt : tp.nPrompt) {
                if (nPrompt == 0) {
                    continue;
                }
                CommandParameters tmpParam;
                tmpParam.model = m;
                tmpParam.nPrompt = nPrompt;
                tmpParam.nGenerate = 0;
                tmpParam.threads = nt;
                tmpParam.useMmap = mmp;
                tmpParam.backend = backend;
                tmpParam.power = power;
                tmpParam.precision = precision;
                tmpParam.memory = memory;
                tmpParam.dynamicOption = dyop;
                tmpParam.nRepeat = tp.nRepeat[0];
                tmpParam.kvCache = "false";
                auto instance = commandParametersInstance(tmpParam);
                instances.push_back(instance);
            }
            for (const auto & nGenerate: tp.nGenerate) {
                CommandParameters tmpParam;
                tmpParam.model = m;
                tmpParam.nPrompt = 0;
                tmpParam.nGenerate = nGenerate;
                tmpParam.threads = nt;
                tmpParam.useMmap = mmp;
                tmpParam.backend = backend;
                tmpParam.power = power;
                tmpParam.precision = precision;
                tmpParam.memory = memory;
                tmpParam.dynamicOption = dyop;
                tmpParam.nRepeat = tp.nRepeat[0];
                tmpParam.kvCache = "false";
                auto instance = commandParametersInstance(tmpParam);
                instances.push_back(instance);
            }
            for (const auto & nPrompGen : tp.nPrompGen) {
                if (nPrompGen.first == 0 && nPrompGen.second == 0) {
                    continue;
                }
                CommandParameters tmpParam;
                tmpParam.model = m;
                tmpParam.nPrompt = nPrompGen.first;
                tmpParam.nGenerate = nPrompGen.second;
                tmpParam.threads = nt;
                tmpParam.useMmap = mmp;
                tmpParam.backend = backend;
                tmpParam.power = power;
                tmpParam.precision = precision;
                tmpParam.memory = memory;
                tmpParam.dynamicOption = dyop;
                tmpParam.nRepeat = tp.nRepeat[0];
                tmpParam.kvCache = "false";
                auto instance = commandParametersInstance(tmpParam);
                instances.push_back(instance);
            }
        }

    return instances;
}

std::string getDirectoryOf(const std::string& file_path, std::string& modelname) {
    // weight filename
    std::string weight_name = "llm.mnn.weight";
    std::ifstream file(file_path.c_str());
    std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    rapidjson::Document doc;
    doc.Parse(json_str.c_str());

    if (doc.HasMember("llm_weight") && doc["llm_weight"].IsString()) {
        weight_name = doc["llm_weight"].GetString();
    }

    size_t pos = file_path.find_last_of("/\\");
    if (pos == std::string::npos) {
        MNN_ERROR("Invalid model config path\n");
        return "";
    }
    auto dir = file_path.substr(0, pos);
    pos = dir.find_last_of("/\\");
    modelname = dir.substr(pos + 1, -1);
    return MNNFilePathConcat(dir, weight_name);
}

static void printUsage(int /* argc */, char ** argv) {
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help\n");
    printf("  -m, --model <filename>                    (default: %s)\n", join(runtimeParamsDefaults.model, ",").c_str());
    printf("  -a, --backends <cpu,opencl,metal>         (default: %s)\n", "cpu");
    printf("  -c, --precision <n>                       (default: %s) | Note: (0:Normal(for cpu bakend, 'Nornal' is 'High'),1:High,2:Low)\n", join(runtimeParamsDefaults.precision, ",").c_str());
    printf("  -t, --threads <n>                         (default: %s)\n", join(runtimeParamsDefaults.threads, ",").c_str());
    printf("  -p, --n-prompt <n>                        (default: %s)\n", join(testParamsDefaults.nPrompt, ",").c_str());
    printf("  -n, --n-gen <n>                           (default: %s)\n", join(testParamsDefaults.nGenerate, ",").c_str());
    printf("  -pg <pp,tg>                               (default: %s)\n", join(transform2String(testParamsDefaults.nPrompGen, pairString), ",").c_str());
    printf("  -mmp, --mmap <0|1>                        (default: %s)\n", join(runtimeParamsDefaults.useMmap, ",").c_str());
    printf("  -rep, --n-repeat <n>                      (default: %s)\n", join(testParamsDefaults.nRepeat, ",").c_str());
    printf("  -kv, --kv-cache <true|false>              (default: %s) | Note: if true: Every time the LLM model generates a new word, it utilizes the cached KV-cache\n", "false");
    printf("  -fp, --file-print <stdout|filename>       (default: %s)\n", "stdout");
}


static bool parseCmdParams(int argc, char ** argv, RuntimeParameters & runtimeParams, TestParameters & testParams, FILE** outfile) {
    std::string       arg;
    bool              invalidParam = false;
    const std::string argPrefix    = "--";
    const char        splitDelim   = ',';

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, argPrefix.size(), argPrefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "-h" || arg == "--help") {
            printUsage(argc, argv);
            return true;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<std::string>(argv[i], splitDelim);
            runtimeParams.model.insert(runtimeParams.model.end(), p.begin(), p.end());
        } else if (arg == "-p" || arg == "--n-prompt") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            testParams.nPrompt.insert(testParams.nPrompt.end(), p.begin(), p.end());
        } else if (arg == "-n" || arg == "--n-gen") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            testParams.nGenerate.insert(testParams.nGenerate.end(), p.begin(), p.end());
        } else if (arg == "-pg") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<std::string>(argv[i], ',');
            if (p.size() != 2) {
                invalidParam = true;
                break;
            }
            testParams.nPrompGen.push_back({ std::stoi(p[0]), std::stoi(p[1]) });
        } else if (arg == "-a" || arg == "--backends") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto ba = splitString<std::string>(argv[i], splitDelim);
            std::vector<int> p;
            for (auto& type: ba) {
                if (type == "metal") {
                    p.emplace_back(1);
                } else if (type == "opencl") {
                    p.emplace_back(3);
                } else {
                    p.emplace_back(0);
                }
            }
            runtimeParams.backends.insert(runtimeParams.backends.end(), p.begin(), p.end());
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            runtimeParams.threads.insert(runtimeParams.threads.end(), p.begin(), p.end());
        } else if (arg == "-mmp" || arg == "--mmap") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<bool>(argv[i], splitDelim);
            runtimeParams.useMmap.insert(runtimeParams.useMmap.end(), p.begin(), p.end());
        } else if (arg == "-c" || arg == "--precision") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            runtimeParams.precision.insert(runtimeParams.precision.end(), p.begin(), p.end());
        } else if (arg == "--memory") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            runtimeParams.memory.insert(runtimeParams.memory.end(), p.begin(), p.end());
        } else if (arg == "--power") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            runtimeParams.power.insert(runtimeParams.power.end(), p.begin(), p.end());
        } else if (arg == "-dyo" || arg == "--dynamicOption") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            runtimeParams.dynamicOption.insert(runtimeParams.dynamicOption.end(), p.begin(), p.end());
        } else if (arg == "-rep" || arg == "--n-repeat") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            testParams.nRepeat.insert(testParams.nRepeat.end(), p.begin(), p.end());
        } else if (arg == "-kv" || arg == "--kv-cache") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<std::string>(argv[i], splitDelim);
            testParams.kvCache.insert(testParams.kvCache.end(), p.begin(), p.end());
        } else if (arg == "-fp" || arg == "--file-print") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<std::string>(argv[i], splitDelim);
            if (!MNNFileExist(p[0].c_str())) {
                MNNCreateFile(p[0].c_str());
            }
            *outfile = openFile(p[0].c_str(), false);
        } 
        else {
            invalidParam = true;
            break;
        }
    } // parse end


    if (invalidParam) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        printUsage(argc, argv);
        return false;
    }

    // set defaults
    if (runtimeParams.model.empty()) {
        runtimeParams.model = runtimeParamsDefaults.model;
    }
    if (testParams.nPrompt.empty()) {
        testParams.nPrompt = testParamsDefaults.nPrompt;
    }
    if (testParams.nGenerate.empty()) {
        testParams.nGenerate = testParamsDefaults.nGenerate;
    }
    if (testParams.nPrompGen.empty()) {
        testParams.nPrompGen = testParamsDefaults.nPrompGen;
    }
    if (runtimeParams.useMmap.empty()) {
        runtimeParams.useMmap = runtimeParamsDefaults.useMmap;
    }
    if (runtimeParams.backends.empty()) {
        runtimeParams.backends = runtimeParamsDefaults.backends;
    }
    if (runtimeParams.memory.empty()) {
        runtimeParams.memory = runtimeParamsDefaults.memory;
    }
    if (runtimeParams.precision.empty()) {
        runtimeParams.precision = runtimeParamsDefaults.precision;
    }
    if (runtimeParams.power.empty()) {
        runtimeParams.power = runtimeParamsDefaults.power;
    }
    if (runtimeParams.threads.empty()) {
        runtimeParams.threads = runtimeParamsDefaults.threads;
    }
    if (runtimeParams.dynamicOption.empty()) {
        runtimeParams.dynamicOption = runtimeParamsDefaults.dynamicOption;
    }
    if (testParams.nRepeat.empty()) {
        testParams.nRepeat = testParamsDefaults.nRepeat;
    }
    if (testParams.kvCache.empty()) {
        testParams.kvCache = testParamsDefaults.kvCache;
    }

    return true;
}


static Llm* buildLLM(const std::string& config_path, int backend, int memory, int precision, int threads, int power, int dynamic_option, bool use_mmap) {
    auto llmPtr = Llm::createLLM(config_path);
    std::map<int, std::string> lever = {{0,"normal"}, {1, "high"}, {2, "low"}};
    std::map<int, std::string> backend_type = {{0, "cpu"}, {1, "metal"}, {3, "opencl"}};
    std::map<bool, std::string> mmap = {{true,"true"}, {false, "false"}};

    bool setSuccess = true;
    setSuccess &= llmPtr->set_config("{\"precision\":\"" + lever[precision] + "\"}");
    if (!setSuccess) {
        MNN_ERROR("precison for LLM config set error\n");
        return nullptr;
    }
    setSuccess &= llmPtr->set_config("{\"memory\":\"" + lever[memory] + "\"}");
    if (!setSuccess) {
        MNN_ERROR("memory for LLM config set error\n");
        return nullptr;
    }
    setSuccess &= llmPtr->set_config("{\"power\":\"" + lever[power] + "\"}");
    if (!setSuccess) {
        MNN_ERROR("power for LLM config set error\n");
        return nullptr;
    }
    setSuccess &= llmPtr->set_config("{\"backend_type\":\"" + backend_type[backend] + "\"}");
    if (!setSuccess) {
        MNN_ERROR("backend_type for LLM config set error\n");
        return nullptr;
    }
    setSuccess &= llmPtr->set_config("{\"thread_num\":" + std::to_string(threads) + "}");
    if (!setSuccess) {
        MNN_ERROR("thread_num for LLM config set error\n");
        return nullptr;
    }
    setSuccess &= llmPtr->set_config("{\"dynamic_option\":" + std::to_string(dynamic_option) + "}");
    if (!setSuccess) {
        MNN_ERROR("dynamic_option for LLM config set error\n");
        return nullptr;
    }
    setSuccess &= llmPtr->set_config("{\"use_mmap\":" + mmap[use_mmap] + "}");
    if (!setSuccess) {
        MNN_ERROR("use_mmap for LLM config set error\n");
        return nullptr;
    }
    setSuccess &= llmPtr->set_config("{\"tmp_path\":\"tmp\"}");
    return llmPtr;
}

int main(int argc, char ** argv) {
    RuntimeParameters runtimeParams;
    TestParameters testParams;
    FILE* outfile = stdout;
    bool parseSuccess = parseCmdParams(argc, argv, runtimeParams, testParams, &outfile);
    if (!parseSuccess) {
        MNN_ERROR("Parse arguments error\n");
        return -1;
    }
    std::vector<commandParametersInstance> paramsInstances = get_cmd_params_instances(runtimeParams, testParams);
    std::unique_ptr<Printer> printer_(new markdownPrinter());
    bool printHeader = true;

    for (const auto & instance: paramsInstances) {
        TestInstance t(instance);
        auto llmWeightPath = getDirectoryOf(t.modelConfigFile, t.modelType); // To check path

        file_t file = MNNOpenFile(llmWeightPath.c_str(), MNN_FILE_READ);
        t.modelSize = MNNGetFileSize(file);

        MNN::BackendConfig backendConfig;
        auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
        MNN::Express::ExecutorScope scope(executor);

        auto llmPtr = buildLLM(instance.mCmdParam.model, instance.mCmdParam.backend, instance.mCmdParam.memory, instance.mCmdParam.precision, instance.mCmdParam.threads, instance.mCmdParam.power, instance.mCmdParam.dynamicOption, instance.mCmdParam.useMmap);
        std::unique_ptr<Llm> llm(llmPtr);
        llm->load();
        auto context = llm->getContext();
        if (instance.mCmdParam.nGenerate > 0) {
            llm->set_config("{\"max_new_tokens\":1}");
        }
        
        auto prompt_tokens = instance.mCmdParam.nPrompt;
        auto decodeTokens = instance.mCmdParam.nGenerate;

        // llm_demo test
        if (instance.mCmdParam.kvCache == "true") {
            std::vector<int> tokens(prompt_tokens, 16);
            
            for (int i = 0; i < instance.mCmdParam.nRepeat + 1; ++i) {
                llm->response(tokens, nullptr, nullptr, decodeTokens);
                auto prefillTime = context->prefill_us;
                auto decodeTime = context->decode_us;
                if (i > 0) { // Exclude the first performance value.
                    t.prefillUs.push_back(prefillTime);
                    t.decodeUs.push_back(decodeTime);
                }
            }
            if (printHeader) {
                printer_->fout = outfile;
                printer_->printHeader(runtimeParams, testParams);
                printHeader = false;
            }
            printer_->printPerformance(t);
            // Cool
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        
        // llama.cpp llama-bench test
        if (instance.mCmdParam.kvCache == "false") {
            int tok = 16;
            std::vector<int> tokens(prompt_tokens, tok);
            std::vector<int> tokens1(1, tok);

            for (int i = 0; i < instance.mCmdParam.nRepeat + 1; ++i) {
                int64_t sampler_us = 0;
                if (prompt_tokens) {
                    llm->response(tokens, nullptr, nullptr, 1);
                    sampler_us += context->prefill_us;
                }
                if (decodeTokens) {
                    auto rep = decodeTokens;
                    while (rep--) {
                        llm->response(tokens1, nullptr, nullptr, 1);
                        sampler_us += context->prefill_us;
                    }
                }
                if (i > 0) {
                    t.samplesUs.push_back(sampler_us);
                }
            }
            
            if (printHeader) {
                printer_->fout = outfile;
                printer_->printHeader(runtimeParams, testParams);
                printHeader = false;
            }
            printer_->printPerformance(t);
            // Cool
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            
        }
    }

    fprintf(printer_->fout, "\n");
    if (printer_->fout != stdout) {
        fclose(printer_->fout);
    }
    return 0;
}
