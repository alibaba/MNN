#pragma once

#include <string>

// Read-only mmap
class FileMemMapper {
public:
    explicit FileMemMapper(const std::string& path);

    // Move ctor
    explicit FileMemMapper(FileMemMapper&& other);

    ~FileMemMapper();

    // Returns true if the file is valid, false if otherwise.
    explicit operator bool() const;

    // Returns true if the file is valid, false if otherwise.
    bool valid() const;

    // Get the file buffer and its corresponding size in bytes
    std::pair<char*, size_t> get() const;

    // Get the file buffer address
    void* getAddr() const;

    // Get the file size in bytes
    size_t getSize() const;

private:
    int mFd = -1;
    void* mBuffer = nullptr;
    size_t mSize = 0;
};