#pragma once

#include "core/MNNFileUtils.h"
#include <string>

namespace MNN {
// Snapshot files are independent of writable KV mappings; publish a manifest only after all layers succeed.
inline bool writePrefixSnapshot(const std::string& path, const void* data, size_t bytes) {
    if (!data || bytes == 0) return false;
    const auto file = MNNCreateFile(path.c_str());
    if (file == INVALID_FILE) return false;
    const bool written = MNNWriteFile(file, const_cast<void*>(data), bytes) == bytes;
    const bool closed = MNNCloseFile(file) == NO_ERROR;
    if (!written || !closed) MNNRemoveFile(path.c_str());
    return written && closed;
}
}
