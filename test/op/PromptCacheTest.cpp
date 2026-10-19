//
//  PromptCacheTest.cpp
//  MNN
//
//  Tests for prompt cache utilities.
//

#include <MNN/MNNDefine.h>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "MNNTestSuite.h"
#include "core/MNNFileUtils.h"
#include "prompt_cache_utils.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

using MNN::Transformer::stripThinkBlocks;

class PromptCacheStripThinkTest : public MNNTestCase {
public:
    virtual ~PromptCacheStripThinkTest() = default;
    virtual bool run(int precision) {
        // Test 1: basic strip
        {
            std::string text = "Hello <think>reasoning here</think> world";
            stripThinkBlocks(text);
            MNNTEST_ASSERT(text == "Hello  world");
        }
        // Test 2: multiple blocks
        {
            std::string text = "<think>a</think>mid<think>b</think>end";
            stripThinkBlocks(text);
            MNNTEST_ASSERT(text == "midend");
        }
        // Test 3: no tags
        {
            std::string text = "no tags here";
            stripThinkBlocks(text);
            MNNTEST_ASSERT(text == "no tags here");
        }
        // Test 4: trailing newlines after </think>
        {
            std::string text = "before<think>x</think>\n\nafter";
            stripThinkBlocks(text);
            MNNTEST_ASSERT(text == "beforeafter");
        }
        // Test 5: unclosed <think> (no </think>)
        {
            std::string text = "before<think>unclosed";
            stripThinkBlocks(text);
            MNNTEST_ASSERT(text == "before<think>unclosed");
        }
        // Test 6: empty string
        {
            std::string text = "";
            stripThinkBlocks(text);
            MNNTEST_ASSERT(text == "");
        }
        // Test 7: trailing \r\n mix
        {
            std::string text = "a<think>b</think>\r\nc";
            stripThinkBlocks(text);
            MNNTEST_ASSERT(text == "ac");
        }
        return true;
    }
};
MNNTestSuiteRegister(PromptCacheStripThinkTest, "op/prompt_cache");

#if defined(__unix__) || defined(__APPLE__)
namespace {

class PromptCacheFilesFixture {
public:
    PromptCacheFilesFixture() {
        const char* tmp = std::getenv("TMPDIR");
#ifdef __ANDROID__
        const char* fallback = "/data/local/tmp";
#else
        const char* fallback = "/tmp";
#endif
        const auto pattern = MNNFilePathConcat(tmp && tmp[0] == '/' ? tmp : fallback, "mnn_prompt_cache_XXXXXX");
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        if (::mkdtemp(buffer.data()) != nullptr) {
            mDirectory = buffer.data();
        }
    }

    ~PromptCacheFilesFixture() {
        if (mDirectory.empty()) {
            return;
        }
        // Remove only files created by this fixture, even on an assertion failure.
        for (const auto& file : mFiles) {
            if (MNNFileExist(file.c_str()) && MNNRemoveFile(file.c_str()) != MNN::NO_ERROR) {
                MNN_ERROR("PromptCacheFiles: cannot remove %s\n", file.c_str());
            }
        }
        if (::rmdir(mDirectory.c_str()) != 0) {
            MNN_ERROR("PromptCacheFiles: cannot remove directory %s\n", mDirectory.c_str());
        }
    }

    bool ready() const {
        return !mDirectory.empty();
    }

    bool write(const std::string& name, const void* data, size_t size) {
        if (!ready()) {
            return false;
        }
        const auto filePath = path(name);
        auto file = std::fopen(filePath.c_str(), "wb");
        if (file == nullptr) {
            return false;
        }
        mFiles.insert(filePath);
        const bool written = size == 0 || std::fwrite(data, 1, size, file) == size;
        const bool closed = std::fclose(file) == 0;
        return written && closed;
    }

    bool writeData(const std::string& name, size_t size) {
        const std::vector<unsigned char> bytes(size, 0x5a);
        return write(name, bytes.data(), bytes.size());
    }

    bool remove(const std::string& name) {
        return MNNRemoveFile(path(name).c_str()) == MNN::NO_ERROR;
    }

    bool writeModel(const MNN::NetT& net) {
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(MNN::Net::Pack(builder, &net));
        flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
        MNNTEST_ASSERT(MNN::VerifyNetBuffer(verifier));
        return write("model.mnn", builder.GetBufferPointer(), builder.GetSize());
    }

    bool valid(int layerCount = 3) const {
        return MNN::Transformer::validatePrefixCacheFiles(path("model.mnn"), path("prefix"), layerCount);
    }

private:
    std::string path(const std::string& name) const {
        return MNNFilePathConcat(mDirectory, name);
    }

    std::string mDirectory;
    std::set<std::string> mFiles;
};

void addCacheAttention(MNN::NetT& net, bool kvCache = true) {
    std::unique_ptr<MNN::OpT> op(new MNN::OpT);
    op->type = MNN::OpType_Attention;
    op->main.type = MNN::OpParameter_AttentionParam;
    op->main.value = new MNN::AttentionParamT;
    op->main.AsAttentionParam()->kv_cache = kvCache;
    net.oplists.emplace_back(std::move(op));
}

void addCacheLinearAttention(MNN::NetT& net, const char* type) {
    std::unique_ptr<MNN::OpT> op(new MNN::OpT);
    op->type = MNN::OpType_LinearAttention;
    op->main.type = MNN::OpParameter_LinearAttentionParam;
    op->main.value = new MNN::LinearAttentionParamT;
    op->main.AsLinearAttentionParam()->attn_type = type;
    net.oplists.emplace_back(std::move(op));
}

} // namespace
#endif

class PromptCacheFilesTest : public MNNTestCase {
public:
    bool run(int precision) override {
#if defined(__unix__) || defined(__APPLE__)
        PromptCacheFilesFixture files;
        MNNTEST_ASSERT(files.ready());
        MNN::NetT mixed;
        // Non-caching Attention must not consume a prefix-cache layer index.
        addCacheAttention(mixed, false);
        addCacheAttention(mixed);
        addCacheLinearAttention(mixed, "short_conv");
        addCacheLinearAttention(mixed, "gated_delta_rule");
        MNNTEST_ASSERT(files.writeModel(mixed));

        // Cache payload sizes may differ across layers and between K and V.
        const size_t keySizes[] = {32, 7, 64};
        const size_t valueSizes[] = {48, 0, 96};
        for (int layer = 0; layer < 3; ++layer) {
            const auto base = "prefix_" + std::to_string(layer);
            MNNTEST_ASSERT(files.writeData(base + ".k", keySizes[layer]));
            MNNTEST_ASSERT(files.writeData(base + "_sync.k", 0));
            if (valueSizes[layer] != 0) {
                MNNTEST_ASSERT(files.writeData(base + ".v", valueSizes[layer]));
                MNNTEST_ASSERT(files.writeData(base + "_sync.v", 0));
            }
        }
        MNNTEST_ASSERT(files.valid());

        // Both files absent must fail independently for Attention and gated_delta_rule.
        // Restoring either one alone still leaves an invalid marker/data pair.
        for (int layer : {0, 2}) {
            const auto base = "prefix_" + std::to_string(layer);
            MNNTEST_ASSERT(files.remove(base + ".v"));
            MNNTEST_ASSERT(files.remove(base + "_sync.v"));
            MNNTEST_ASSERT(!files.valid());
            MNNTEST_ASSERT(files.writeData(base + ".v", valueSizes[layer]));
            MNNTEST_ASSERT(!files.valid());
            MNNTEST_ASSERT(files.writeData(base + "_sync.v", 0));
            MNNTEST_ASSERT(files.valid());
            MNNTEST_ASSERT(files.remove(base + ".v"));
            MNNTEST_ASSERT(!files.valid());
            MNNTEST_ASSERT(files.writeData(base + ".v", valueSizes[layer]));
            MNNTEST_ASSERT(files.valid());

            MNNTEST_ASSERT(files.writeData(base + ".v", 0));
            MNNTEST_ASSERT(!files.valid());
            MNNTEST_ASSERT(files.writeData(base + ".v", valueSizes[layer]));
            MNNTEST_ASSERT(files.valid());
        }

        // Every layer, including short_conv, needs nonempty K and its sync marker.
        for (int layer = 0; layer < 3; ++layer) {
            const auto base = "prefix_" + std::to_string(layer);
            MNNTEST_ASSERT(files.remove(base + ".k"));
            MNNTEST_ASSERT(!files.valid());
            MNNTEST_ASSERT(files.writeData(base + ".k", keySizes[layer]));
            MNNTEST_ASSERT(files.valid());
            MNNTEST_ASSERT(files.remove(base + "_sync.k"));
            MNNTEST_ASSERT(!files.valid());
            MNNTEST_ASSERT(files.writeData(base + "_sync.k", 0));
            MNNTEST_ASSERT(files.valid());
            MNNTEST_ASSERT(files.writeData(base + ".k", 0));
            MNNTEST_ASSERT(!files.valid());
            MNNTEST_ASSERT(files.writeData(base + ".k", keySizes[layer]));
            MNNTEST_ASSERT(files.valid());
        }

        // short_conv requires K only: stray V data, marker, or both are mismatches.
        MNNTEST_ASSERT(files.writeData("prefix_1.v", 17));
        MNNTEST_ASSERT(!files.valid());
        MNNTEST_ASSERT(files.writeData("prefix_1_sync.v", 0));
        MNNTEST_ASSERT(!files.valid());
        MNNTEST_ASSERT(files.remove("prefix_1.v"));
        MNNTEST_ASSERT(!files.valid());
        MNNTEST_ASSERT(files.remove("prefix_1_sync.v"));
        MNNTEST_ASSERT(files.valid());

        for (int count : {-1, 0, 2, 4}) {
            MNNTEST_ASSERT(!files.valid(count));
        }
        MNNTEST_ASSERT(files.valid());

        // Valid FlatBuffers can still describe unsupported cache layouts.
        for (const char* type : {"unknown_attention", ""}) {
            mixed.oplists.back()->main.AsLinearAttentionParam()->attn_type = type;
            MNNTEST_ASSERT(files.writeModel(mixed));
            MNNTEST_ASSERT(!files.valid());
            mixed.oplists.back()->main.AsLinearAttentionParam()->attn_type = "gated_delta_rule";
            MNNTEST_ASSERT(files.writeModel(mixed));
            MNNTEST_ASSERT(files.valid());
        }
        mixed.subgraphs.emplace_back(new MNN::SubGraphProtoT);
        MNNTEST_ASSERT(files.writeModel(mixed));
        MNNTEST_ASSERT(!files.valid());
        mixed.subgraphs.clear();
        MNNTEST_ASSERT(files.writeModel(mixed));
        MNNTEST_ASSERT(files.valid());

        for (auto type : {MNN::OpType_Attention, MNN::OpType_LinearAttention}) {
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type = type; // Missing the required parameter table.
            mixed.oplists.emplace_back(std::move(op));
            MNNTEST_ASSERT(files.writeModel(mixed));
            MNNTEST_ASSERT(!files.valid());
            mixed.oplists.pop_back();
            MNNTEST_ASSERT(files.writeModel(mixed));
            MNNTEST_ASSERT(files.valid());
        }
        MNN::NetT empty;
        MNNTEST_ASSERT(files.writeModel(empty));
        MNNTEST_ASSERT(!files.valid());
        MNNTEST_ASSERT(files.writeModel(mixed));
        MNNTEST_ASSERT(files.valid());

        // Exercise actual file loading and FlatBuffer verification, not just metadata.
        const unsigned char malformed[] = {0xff, 0xff, 0xff, 0x7f, 0, 0, 0, 0};
        MNNTEST_ASSERT(files.write("model.mnn", malformed, sizeof(malformed)));
        MNNTEST_ASSERT(!files.valid());
        MNNTEST_ASSERT(files.writeModel(mixed));
        MNNTEST_ASSERT(files.valid());
        MNNTEST_ASSERT(files.writeData("model.mnn", 0));
        MNNTEST_ASSERT(!files.valid());
        MNNTEST_ASSERT(files.writeModel(mixed));
        MNNTEST_ASSERT(files.valid());
        MNNTEST_ASSERT(files.remove("model.mnn"));
        MNNTEST_ASSERT(!files.valid());
        MNNTEST_ASSERT(files.writeModel(mixed));
        MNNTEST_ASSERT(files.valid());

        // Pure Attention also allows heterogeneous nonzero cache shapes/sizes.
        MNN::NetT attention;
        for (int layer = 0; layer < 3; ++layer) {
            addCacheAttention(attention);
        }
        MNNTEST_ASSERT(files.writeModel(attention));
        MNNTEST_ASSERT(!files.valid()); // Former short_conv layer now requires V.
        MNNTEST_ASSERT(files.writeData("prefix_1.v", 17));
        MNNTEST_ASSERT(!files.valid());
        MNNTEST_ASSERT(files.writeData("prefix_1_sync.v", 0));
        MNNTEST_ASSERT(files.valid());
#else
        MNN_PRINT("Skip op/prompt_cache_files: isolated fixtures require Unix mkdtemp\n");
#endif
        return true;
    }
};
MNNTestSuiteRegister(PromptCacheFilesTest, "op/prompt_cache_files");
