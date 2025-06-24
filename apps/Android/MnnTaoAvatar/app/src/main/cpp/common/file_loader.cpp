#include "file_loader.hpp"
#include <iostream>
#include <fstream>
#include "mh_log.hpp"

RawBuffer::RawBuffer(size_t length) {
    mBuffer = malloc(length);
    mLength = length;
}

RawBuffer::~RawBuffer() {
    if (mBuffer != nullptr) {
        free(mBuffer);
    }
}

file_loader::file_loader(const char *fileName) {
#if defined(_MSC_VER)
    wchar_t wFileName[1024];
    if (MultiByteToWideChar(CP_ACP, 0, file, -1, wFileName, sizeof(wFileName)) == 0) {
        mFile = nullptr;
        return;
    }
#if _MSC_VER >= 1400
    if (0 != _wfopen_s(&mFile, wFileName, L"rb")) {
        mFile = nullptr;
        return;
    }
#else
    mFile = _wfopen(wFileName, L"rb");
#endif
#else
    mFile = fopen(fileName, "rb");
#endif
    mFileName = fileName;
    if (mFile == nullptr) {
        MH_ERROR("Failed to open file %s.", mFileName);
    }
}

file_loader::~file_loader() {
    if (mFile != nullptr) {
        fclose(mFile);
    }
}
std::shared_ptr<RawBuffer> file_loader::read(size_t offset, size_t length) {
    if (mFile == nullptr) {
        return nullptr;
    }

    if (length == 0 && offset == 0) {
        fseek(mFile, 0, SEEK_END);
        length = ftell(mFile);
    }

    if (length == 0) {
        return nullptr;
    }

    std::shared_ptr<RawBuffer> buffer(new RawBuffer(length));
    if (!buffer->valid()) {
        MH_ERROR("Failed to alloc memory.");
        return nullptr;
    }

    fseek(mFile, offset, SEEK_SET);
    auto block = reinterpret_cast<char *>(buffer->buffer());
    auto size  = fread(block, 1, gCacheSize, mFile);
    auto currentSize = size;
    block += size;

    while (size == gCacheSize) {
        if (gCacheSize > length - currentSize) {
            int remained = length - currentSize;
            size = fread(block, 1, remained, mFile);
        } else {
            size = fread(block, 1, gCacheSize, mFile);
        }
        if (size > gCacheSize) {
            MH_ERROR("Failed to read file %s.", mFileName);
            return nullptr;
        }
        currentSize += size;
        block+=size;
    }

    int err = ferror(mFile);
    if (err) {
        MH_ERROR("Failed to read file.");
    }
    return buffer;
}

std::shared_ptr<RawBuffer> file_loader::read() {
    return read(0, 0);
}
