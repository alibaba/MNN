#ifndef FILELOADER_HPP
#define FILELOADER_HPP

#include <stdlib.h>
#include <memory>
#include <string>
#if defined(_MSC_VER)
#include "Windows.h"
#endif

class RawBuffer {
public:
    RawBuffer(size_t length);
    ~RawBuffer();

    bool valid() const {
        return mBuffer != nullptr;
    }

    inline size_t length() const {
        return mLength;
    }

    void* buffer() const {
        return mBuffer;
    }
private:
    void *mBuffer = nullptr;
    size_t mLength = 0;
};

class file_loader {
public:
    file_loader(const char *fileName);
    ~file_loader();

    std::shared_ptr<RawBuffer> read();
    std::shared_ptr<RawBuffer> read(size_t offset, size_t length);

    bool valid() const {
        return mFile != nullptr;
    }
private:
    FILE *mFile = nullptr;
    static const int gCacheSize = 4096;
    const char *mFileName = nullptr;
};

#endif //FILELOADER_HPP
