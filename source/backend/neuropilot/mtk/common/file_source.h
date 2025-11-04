#pragma once

#include "cpp11_compat.h"
#include "file_mem_mapper.h"

#include <ostream>
#include <string>

// A file path or buffer wrapper, which is initialized with either a file path or buffer,
// but only allows accessing the file data directly without its path.
//
// If it is initialized with a file path, then the FileSource instance owns and manages the
// lifecycle of the file buffer, and the file will be lazy loaded only when needed unless load() is
// explicitly called.
//
// However, if it is initialized with user defined file buffer, the FileSource instance does not
// manage the lifecycle of the buffer.
//
// NOTE: It is recommeded to pass FileSource by value so that the reference counter in shared_ptr
// works together with RAII, so that the underlying object is destroyed when no one is using it.
class FileSource {
public:
    // Empty file source
    FileSource() {}

    // File source with path
    FileSource(const std::string& path);

    // File source directly with buffer, with optional name as description
    FileSource(const void* data, const size_t size, const std::string& name = "");

    FileSource& operator=(const FileSource& other) {
        mPath = other.mPath;
        mName = other.mName;
        mFileData = other.mFileData;
        mFileMemMapper = other.mFileMemMapper;
        return *this;
    }

    bool operator==(const FileSource& other) const {
        if (empty() && other.empty()) {
            return true;
        }
        if (!mPath.empty()) {
            return mPath == other.mPath;
        } else {
            return (mFileData.data() == other.mFileData.data())
                   && (mFileData.size() == other.mFileData.size());
        }
    }

    // Check if the file source is used, aka not empty
    explicit operator bool() const;

    // Check if the file source is empty
    bool empty() const;

    // Return the path if possible, otherwise return the given name.
    const std::string& getName() const;

    // Get the file buffer. Will load the file if not yet loaded.
    const char* getData() const;

    // Get the file size. Will load the file if not yet loaded.
    size_t getSize() const;

    // Get the file buffer and its size in bytes. Will load the file if not yet loaded.
    std::pair<const char*, size_t> get() const;

    // Returns whether the file can be read successfully
    bool valid() const;

    // Check if the FileSource instance owns and manages the buffer lifecycle
    bool hasBufferOwnership() const;

    // Load the file if not yet loaded if it has path given.
    // Returns true if file is loaded successfully, false if otherwise.
    bool load();

    // Use Direct IO to load the file to the specified destination buffer if loading from file path.
    // Otherwise, a simple memory copy will be performed to copy the data from the source buffer
    // to the destination buffer.
    //
    // If `allowFallback` argument is set to true, it will fallback to using mmap approach when
    // Direct IO fails.
    //
    // To prevent mmap from being invoked, avoid calling these APIs: `getData()`, `get()`, `load()`.
    bool directRead(void* dstBuffer, const size_t dstSize, const bool allowFallback = true) const;

    // Hint for file release to indicate that this instance has done reading the file.
    // Returns true if releasable, false if otherwise.
    // Note that the file will only be released when the last FileMemMapper shared_ptr object that
    // owns the file has been destroyed.
    bool hint_release() const;

private:
    const mtk::cpp11_compat::string_view& getFileData() const;

    void releaseFileData() const;

private:
    std::string mPath;
    std::string mName;
    mutable mtk::cpp11_compat::string_view mFileData;
    mutable std::shared_ptr<FileMemMapper> mFileMemMapper;

    // Used when FileSource is initialized with filepath. This allows file size checking without
    // actually loading or mmap the file.
    const size_t kValidFileSize = 0;
};

// Support FileSource in ostream
std::ostream& operator<<(std::ostream& stream, const FileSource& fileSource);

// Support STL data structures that requires hash
template <>
struct std::hash<FileSource> {
    std::size_t operator()(const FileSource& fileSource) const {
        if (fileSource.hasBufferOwnership()) { // Has access to file path
            return std::hash<std::string>()(fileSource.getName());
        }
        return std::hash<const char*>()(fileSource.getData());
    }
};
