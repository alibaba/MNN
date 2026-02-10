#pragma once

#include <sys/system_properties.h>

#include <string>
#include <vector>

#define DUMP(type)                  DumpHandler::instance().setDumpType(DumpType::type)
#define SHOULD_DUMP(type)           DumpHandler::instance().setDumpType(DumpType::type).isEnabled()
#define SET_DUMP_INDEX(index)       DumpHandler::instance().setIndex(index)
#define SET_DUMP_CHUNK_INDEX(index) DumpHandler::instance().setChunkIndex(index)

enum class DumpType {
    PROMPT,
    RESPONSE,
    INPUTS,
    CHUNK_OUT,
    CACHE
};

class DumpHandler {
private:
    static constexpr char kDumpDir[] = "/data/local/tmp/llm_sdk/dump";

public:
    explicit DumpHandler() {}

    static DumpHandler& instance();

    bool isEnabled() const;

    DumpHandler& setDumpType(const DumpType dumpType);
    DumpHandler& setIndex(const size_t idx);
    DumpHandler& setChunkIndex(const size_t chunkIdx);

    template <typename T>
    void fromValue(const std::string& name, const T& val);

    template <typename T>
    void fromVector(const std::string& name, const std::vector<T>& vec);

    void fromString(const std::string& name, const std::string& str);

    void fromBinary(const std::string& name, const void* buffer, const size_t size);

private:
    std::string getIndexStr() const;

    std::string getChunkName() const;

    std::string getSavePath(const std::string& fileName) const;

private:
    DumpType mDumpType = DumpType::PROMPT;
    size_t mIndex = 0;
    size_t mChunkIndex = 0;
};
