#include "file_source.h"

#include "cpp11_compat.h"
#include "file_direct_reader.h"
#include "logging.h"

#include <sys/stat.h>

#include <ostream>
#include <string>
#include <string_view>

// Get the file size or zero if unable to open the file.
inline size_t getFileSize(const std::string& filepath) {
    struct stat st;
    if (stat(filepath.c_str(), &st) != 0) {
        return 0;
    }
    return st.st_size;
}

FileSource::FileSource(const std::string& path) : mPath(path), kValidFileSize(getFileSize(path)) {}

FileSource::FileSource(const void* data, const size_t size, const std::string& name)
    : mName(name), mFileData((const char*)data, size) {}

// Check if the file source is used, aka not empty
FileSource::operator bool() const {
    return !empty();
}

// Check if the file source is empty
bool FileSource::empty() const {
    return (mPath.empty() && mFileData.empty());
}

// Return the path if possible, otherwise return the given name.
const std::string& FileSource::getName() const {
    static const std::string unnamed = "Unnamed";
    if (!mPath.empty())
        return mPath;
    if (mName.empty())
        return unnamed;
    return mName;
}

const char* FileSource::getData() const {
    if (!valid()) {
        LOG(WARN) << "Unable to load " << *this;
    }
    return getFileData().data();
}

size_t FileSource::getSize() const {
    if (kValidFileSize > 0) {
        return kValidFileSize;
    }
    if (!valid()) {
        LOG(WARN) << "Unable to load " << *this;
    }
    return getFileData().size();
}

std::pair<const char*, size_t> FileSource::get() const {
    if (!valid()) {
        LOG(WARN) << "Unable to load " << *this;
    }
    const auto& fileData = getFileData();
    return {fileData.data(), fileData.size()};
}

// Returns whether the file can be read successfully
bool FileSource::valid() const {
    return (kValidFileSize > 0) || !getFileData().empty();
}

// Check if the FileSource instance owns and manages the buffer lifecycle
bool FileSource::hasBufferOwnership() const {
    return !mPath.empty();
}

// Load the file if not yet loaded and has path given.
// Returns true if file is loaded successfully, false if otherwise.
bool FileSource::load() {
    if (!mFileData.empty() || mPath.empty()) {
        return true;
    }
    DCHECK(!mFileMemMapper);
    mFileMemMapper = std::make_shared<FileMemMapper>(mPath);
    if (mFileMemMapper->valid()) {
        const auto fileData = mFileMemMapper->get();
        const auto data = fileData.first;
        const auto size = fileData.second;
        mFileData = mtk::cpp11_compat::string_view(data, size);
    }
    return mFileMemMapper->valid();
}

bool FileSource::hint_release() const {
    if (!hasBufferOwnership()) {
        return false; // Unable to reopen without having the path
    }
    releaseFileData();
    return true;
}

const mtk::cpp11_compat::string_view& FileSource::getFileData() const {
    const_cast<FileSource*>(this)->load();
    return mFileData;
}

void FileSource::releaseFileData() const {
    if (mFileMemMapper) {
        mFileMemMapper.reset();
    }
    mFileData = mtk::cpp11_compat::string_view();
}

bool FileSource::directRead(void* dstBuffer, const size_t dstSize, const bool allowFallback) const {
    if (!valid()) {
        LOG(WARN) << "Unable to load " << *this;
        return false;
    }

    auto checkSize = [=](const size_t fileSize) {
        if (dstSize < fileSize) {
            LOG(WARN) << "Destination buffer size (" << dstSize
                      << ") is smaller than the file size (" << fileSize << ").";
        }
    };

    if (!mPath.empty()) {
        FileDirectReader directIOLoader(mPath);
        checkSize(directIOLoader.size());
        const auto status = directIOLoader.load(dstBuffer, dstSize);
        if (status) {
            return true; // Successfully loaded using Direct IO
        } else if (!allowFallback) {
            LOG(FATAL) << "Unable to load file using Direct IO: " << *this;
        } else {
            LOG(WARN) << "Failed to load file using Direct IO, falling back to mmap.";
        }
    }

    // Fallback to loading manually using memory copy from file data that may be mmapped
    const auto fileInfo = get();
    const auto fileData = fileInfo.first;
    const auto fileSize = fileInfo.second;
    checkSize(fileSize);
    const auto copySize = std::min(dstSize, fileSize);
    std::memcpy(dstBuffer, fileData, copySize);
    return true;
}

std::ostream& operator<<(std::ostream& stream, const FileSource& fileSource) {
    stream << "<FileSource: " << (fileSource.empty() ? "None" : fileSource.getName()) << ">";
    return stream;
}