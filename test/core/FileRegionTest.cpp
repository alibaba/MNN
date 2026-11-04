#include <MNN/expr/Module.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "core/FileLoader.hpp"
#include <climits>
#include <cstdio>
#include <memory>

#ifndef _WIN32
#include <unistd.h>
#ifdef __ANDROID__
#define MNN_FILE_REGION_TMP "/data/local/tmp/"
#else
#define MNN_FILE_REGION_TMP "/tmp/"
#endif

using namespace MNN;
using namespace MNN::Express;

static bool checkRegions() {
    char path[] = MNN_FILE_REGION_TMP "mnn-file-region-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0)
        return false;
    const char data[] = "prefix123456suffix";
    bool valid = write(fd, data, sizeof(data) - 1) == sizeof(data) - 1;
    const int64_t largeOffset = (int64_t(1) << 32) + 19;
#ifdef __ANDROID__
    valid = valid && lseek64(fd, largeOffset, SEEK_SET) == largeOffset;
#else
    valid = valid && lseek(fd, largeOffset, SEEK_SET) == largeOffset;
#endif
    valid = valid && write(fd, "XY", 2) == 2;
    close(fd);
    {
        const auto locator = MNN::FileLoader::fileRegion(path, 6, 6);
        MNN::FileLoader region(locator.c_str(), true);
        char bytes[7] = {};
        valid = valid && region.valid() && region.read(bytes, 6) && std::string(bytes, 6) == "123456";
        valid = valid && !region.read(bytes, 1) && region.offset(7) != 0 && region.offset(-1) != 0;
        valid = valid && region.offset(3) == 0 && region.read(bytes, 3) && std::string(bytes, 3) == "456";
        MNN::FileLoader reopened(region.path().c_str(), true);
        valid = valid && reopened.valid() && reopened.read(bytes, 6) && std::string(bytes, 6) == "123456";
        MNN::FileLoader bulk(locator.c_str(), true);
        MNN::AutoStorage<uint8_t> buffer;
        valid =
            valid && bulk.read() && bulk.size() == 6 && bulk.merge(buffer) && memcmp(buffer.get(), "123456", 6) == 0;
        valid =
            valid && bulk.offset(0) == 0 && bulk.read() && bulk.merge(buffer) && memcmp(buffer.get(), "123456", 6) == 0;
        const auto large = MNN::FileLoader::fileRegion(path, largeOffset, 2);
        MNN::FileLoader sparse(large.c_str(), true);
        valid = valid && sparse.valid() && sparse.read(bytes, 2) && std::string(bytes, 2) == "XY";
        const auto outside = MNN::FileLoader::fileRegion(path, largeOffset, 3);
        MNN::FileLoader invalid(outside.c_str(), true);
        valid = valid && !invalid.valid();
        MNN::FileLoader malformed("mnn-region:1:999999999999999999999:1:x", true);
        valid = valid && !malformed.valid() && malformed.offset(0) != 0;
    }
    unlink(path);
    return valid;
}

static std::vector<uint8_t> externalGraph(const std::string& path, bool inSubgraph, int64_t weightOffset = 0) {
    flatbuffers::FlatBufferBuilder builder;
    const auto constant = [&](bool externalPath) -> flatbuffers::Offset<MNN::Op> {
        const auto dims = builder.CreateVector(std::vector<int>{1});
        const auto external = builder.CreateVector(std::vector<int64_t>{weightOffset, sizeof(float)});
        MNN::BlobBuilder blob(builder);
        blob.add_dims(dims);
        blob.add_external(external);
        const auto payload = blob.Finish();
        const auto output = builder.CreateVector(std::vector<int>{0});
        const auto file = externalPath ? builder.CreateString(path) : flatbuffers::Offset<flatbuffers::String>();
        MNN::OpBuilder op(builder);
        op.add_type(MNN::OpType_Const);
        op.add_main_type(MNN::OpParameter_Blob);
        op.add_main(payload.Union());
        op.add_outputIndexes(output);
        op.add_externalPath(file);
        return op.Finish();
    };
    const auto main = constant(!inSubgraph && !path.empty());
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<MNN::SubGraphProto>>> subgraphs;
    if (inSubgraph) {
        const auto node = constant(true);
        const auto nodes = builder.CreateVector(std::vector<flatbuffers::Offset<MNN::Op>>{node});
        const auto name = builder.CreateString("unused");
        const auto inputs = builder.CreateVector(std::vector<int>{});
        const auto outputs = builder.CreateVector(std::vector<int>{0});
        const auto tensors =
            builder.CreateVector(std::vector<flatbuffers::Offset<flatbuffers::String>>{builder.CreateString("out")});
        MNN::SubGraphProtoBuilder graph(builder);
        graph.add_name(name);
        graph.add_nodes(nodes);
        graph.add_inputs(inputs);
        graph.add_outputs(outputs);
        graph.add_tensors(tensors);
        const auto subgraph = graph.Finish();
        subgraphs = builder.CreateVector(std::vector<flatbuffers::Offset<MNN::SubGraphProto>>{subgraph});
    }
    const auto ops = builder.CreateVector(std::vector<flatbuffers::Offset<MNN::Op>>{main});
    const auto names =
        builder.CreateVector(std::vector<flatbuffers::Offset<flatbuffers::String>>{builder.CreateString("out")});
    MNN::NetBuilder net(builder);
    net.add_oplists(ops);
    net.add_tensorName(names);
    net.add_outputName(names);
    net.add_tensorNumber(1);
    net.add_subgraphs(subgraphs);
    builder.Finish(net.Finish());
    return std::vector<uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}

static bool checkExternalPaths() {
    char path[] = MNN_FILE_REGION_TMP "mnn-external-path-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0)
        return false;
    const float value = 1.25f;
    const bool written = write(fd, &value, sizeof(value)) == sizeof(value);
    close(fd);
    bool valid = written;
    MNN::ScheduleConfig config;
    config.numThread = 1;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime(
        MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
    if (!runtime) {
        unlink(path);
        return false;
    }
    for (bool subgraph : {false, true}) {
        const auto data = externalGraph(path, subgraph);
        runtime->setExternalFile(path);
        std::unique_ptr<MNN::Express::Module> legacy(
            MNN::Express::Module::load({}, {"out"}, data.data(), data.size(), runtime));
        valid = valid && legacy != nullptr;
        if (legacy) {
            const auto output = legacy->onForward({});
            valid =
                valid && output.size() == 1 && output[0]->readMap<float>() && output[0]->readMap<float>()[0] == value;
        }
        valid = valid && runtime->setExternalFile(path, 0, sizeof(value));
        std::unique_ptr<MNN::Express::Module> bounded(
            MNN::Express::Module::load({}, {"out"}, data.data(), data.size(), runtime));
        valid = valid && bounded == nullptr;
    }
    unlink(path);
    return valid;
}

static bool checkModuleRegions() {
    const auto graph = externalGraph("", false);
    char path[] = MNN_FILE_REGION_TMP "mnn-module-region-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0)
        return false;
    FILE* file = fdopen(fd, "wb");
    if (!file) {
        close(fd);
        unlink(path);
        return false;
    }
    const uint8_t prefix[16] = {};
    const float weight = 1.25f;
    bool valid = fwrite(prefix, 1, sizeof(prefix), file) == sizeof(prefix) &&
                 fwrite(graph.data(), 1, graph.size(), file) == graph.size() &&
                 fwrite(&weight, 1, sizeof(weight), file) == sizeof(weight) && fwrite("suffix", 1, 6, file) == 6;
    fclose(file);
    ScheduleConfig config;
    config.numThread = 1;
    std::shared_ptr<Executor::RuntimeManager> runtime(Executor::RuntimeManager::createRuntimeManager(config));
    if (!runtime) {
        unlink(path);
        return false;
    }
    valid = valid && runtime->setExternalFile(path, sizeof(prefix) + graph.size(), sizeof(weight));
    valid = valid && !runtime->setExternalFile(path, -1, sizeof(weight)) && !runtime->setExternalFile(path, 0, 0) &&
            !runtime->setExternalFile(path, INT64_MAX, 1);
    valid = valid && !runtime->setExternalFile(path, sizeof(prefix) + graph.size(), 100) &&
            !runtime->setExternalFile(std::string(path) + ".missing", 0, sizeof(weight));
    for (bool memory : {false, true}) {
        std::unique_ptr<Module> module(memory ? Module::load({}, {"out"}, graph.data(), graph.size(), runtime)
                                              : Module::load({}, {"out"}, path, sizeof(prefix), graph.size(), runtime));
        valid = valid && module != nullptr;
        if (module) {
            auto result = module->onForward({});
            valid =
                valid && result.size() == 1 && result[0]->readMap<float>() && result[0]->readMap<float>()[0] == weight;
        }
    }
    std::unique_ptr<Module> truncated(Module::load({}, {"out"}, path, sizeof(prefix), graph.size() / 2, runtime));
    std::unique_ptr<Module> outside(Module::load({}, {"out"}, path, sizeof(prefix), graph.size() + 100, runtime));
    valid = valid && !truncated && !outside;
    for (auto range : {std::make_pair(int64_t(-1), int64_t(graph.size())), std::make_pair(int64_t(0), int64_t(0)),
                       std::make_pair(INT64_MAX, int64_t(1))}) {
        std::unique_ptr<Module> invalid(Module::load({}, {"out"}, path, range.first, range.second, runtime));
        valid = valid && !invalid;
    }
    for (int64_t offset : {int64_t(2), int64_t(8)}) {
        const auto invalidGraph = externalGraph("", false, offset);
        std::unique_ptr<Module> invalid(Module::load({}, {"out"}, invalidGraph.data(), invalidGraph.size(), runtime));
        valid = valid && !invalid;
    }
    valid = valid && runtime->setExternalFile(path, sizeof(prefix) + graph.size(), sizeof(weight) - 1);
    std::unique_ptr<Module> shortWeight(Module::load({}, {"out"}, graph.data(), graph.size(), runtime));
    valid = valid && !shortWeight;
    const std::string weightPath = std::string(path) + ".weight";
    valid = valid && FileLoader::write(path, {graph.data(), graph.size()}) &&
            FileLoader::write(weightPath.c_str(), {&weight, sizeof(weight)});
    runtime->setExternalFile("");
    std::unique_ptr<Module> legacy(Module::load({}, {"out"}, path, runtime));
    valid = valid && legacy != nullptr;
    if (legacy) {
        auto result = legacy->onForward({});
        valid = valid && result.size() == 1 && result[0]->readMap<float>() && result[0]->readMap<float>()[0] == weight;
    }
    valid = valid && runtime->setExternalFile(weightPath, 0, sizeof(weight));
    valid = valid && FileLoader::write(weightPath.c_str(), {&weight, sizeof(weight) - 1});
    std::unique_ptr<Module> changedWeight(Module::load({}, {"out"}, graph.data(), graph.size(), runtime));
    valid = valid && !changedWeight;
    unlink(weightPath.c_str());
    std::unique_ptr<Module> missingWeight(Module::load({}, {"out"}, graph.data(), graph.size(), runtime));
    valid = valid && !missingWeight;
    unlink(path);
    return valid;
}

class FileRegionTest : public MNNTestCase {
public:
    bool run(int precision) override { return checkRegions() && checkModuleRegions() && checkExternalPaths(); }
};
MNNTestSuiteRegister(FileRegionTest, "core/file_region");
#endif
