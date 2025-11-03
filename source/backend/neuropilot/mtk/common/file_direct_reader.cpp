#include "file_direct_reader.h"

#include "logging.h"
#include "scope_profiling.h"
#include "thread_pool.h"
#include "timer.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <string>

// This class uses RAII to handle the opening and closing of a file using C POSIX APIs
class CFileGuard {
public:
    CFileGuard(const std::string& filepath, const char* mode = "rb") {
        mFile = std::fopen(filepath.c_str(), mode);
    }

    ~CFileGuard() {
        if (mFile) {
            std::fclose(mFile);
        }
    }

    bool status() const { return mFile != nullptr; }

    std::FILE* file() { return mFile; }

private:
    std::FILE* mFile = nullptr;
};

inline uint32_t nextPow2(uint32_t n) {
    if (n == 0) {
        return 1;
    }
    // Bit-twiddling, see https://stackoverflow.com/a/1322548
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

FileDirectReader::FileDirectReader(const std::string& filepath, const size_t numWorkers)
    : kFilePath(filepath), kNumWorkers(legalizeNumWorkers(numWorkers)), mThreadPool(kNumWorkers) {
    mFileFd = open(filepath.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (mFileFd < 0) {
        LOG(ERROR) << "FileDirectReader failed to open file: " << filepath;
        return;
    }

    struct stat st;
    if (fstat(mFileFd, &st) != 0) {
        LOG(ERROR) << "FileDirectReader failed get file size: " << filepath;
        return;
    }
    mFileSize = st.st_size;

    const size_t numBlocks = mFileSize / kBlockSize; // Floor division
    mWholeBlocksSize = numBlocks * kBlockSize;
    mLastBlockSize = mFileSize % kBlockSize;
    LOG(DEBUG) << "FileDirectReader: fileSize is " << mFileSize
               << " (whole blocks = " << mWholeBlocksSize << ", remainder = " << mLastBlockSize
               << ")";
}

FileDirectReader::~FileDirectReader() {
    if (mFileFd >= 0) {
        close(mFileFd);
    }
}

size_t FileDirectReader::legalizeNumWorkers(const size_t numWorkers) {
    return std::min(static_cast<size_t>(nextPow2(numWorkers)), kBlockSize / kMemoryAlignment);
}

FileDirectReader::operator bool() const {
    return valid();
}

bool FileDirectReader::valid() const {
    return (mFileFd >= 0 && mFileSize > 0);
}

size_t FileDirectReader::size() const {
    return mFileSize;
}

template <size_t kMemoryAlignment>
static void loadFilePart(void* dstBuffer, const size_t dstSize, const int fileFd,
                         const size_t startOffset, const size_t filePartSize,
                         const size_t bufferSize, std::atomic<bool>* status) {

    void* copyBuffer;
    posix_memalign(&copyBuffer, kMemoryAlignment, bufferSize);

    size_t totalReadSize = 0;
    while (totalReadSize < filePartSize) {
        const size_t sizeRemaining = filePartSize - totalReadSize;
        const size_t curReadSize = std::min(bufferSize, sizeRemaining);
        const size_t curOffset = startOffset + totalReadSize;

        const size_t dstSpaceLeft = dstSize - std::min(dstSize, curOffset);
        if (dstSpaceLeft == 0) {
            break;
        }


        const auto actualReadSize = pread(fileFd, copyBuffer, curReadSize, curOffset);

        if (actualReadSize < 0) {
            LOG(ERROR) << "Failed to read file " << strerror(errno);
            LOG(ERROR) << "pread returned " << actualReadSize << " with args: (fd=" << fileFd
                       << ", buf=" << copyBuffer << ", count=" << curReadSize
                       << ", offset=" << (curOffset) << ")";
            free(copyBuffer);
            *status = false;
            return;
        }
        const auto copySize = std::min(dstSpaceLeft, static_cast<size_t>(actualReadSize));
        std::memcpy((char*)dstBuffer + curOffset, copyBuffer, copySize);
        totalReadSize += copySize;
    }
    free(copyBuffer);
}

bool FileDirectReader::load(void* dstBuffer, const size_t dstSize) {
    if (mFileFd < 0) {
        return false;
    }

    if (dstSize < mFileSize) {
        LOG(WARN) << "Destination buffer size (" << dstSize << ") is smaller than the file size ("
                  << mFileSize << ").";
    }

    // Each thread worker reads filePartSize worth of data from the file
    const size_t filePartSize = mWholeBlocksSize / kNumWorkers;
    DCHECK_EQ(mWholeBlocksSize % kNumWorkers, 0);

    constexpr size_t bufferSize = kBlockSize;

    Timer timer;
    timer.start();

    // A failed load status means at least one file part failed to load.
    std::atomic<bool> loadStatus(true);

    size_t offset = 0;
    while (offset < mWholeBlocksSize) {
        // NOTE: `filePartSize` and `offset` have to be multiple of 4096.
        mThreadPool.push(&loadFilePart<kMemoryAlignment>, dstBuffer, dstSize, mFileFd, offset,
                         filePartSize, bufferSize, &loadStatus);
        offset += filePartSize;
    }

    mThreadPool.joinAll();

    if (!loadStatus) {
        LOG(ERROR) << "Failed to read file with Direct IO: " << kFilePath;
        return false;
    }

    const auto asyncElapsed = timer.reset<Timer::ms>();

    // Read remainder tail block
    if (mLastBlockSize > 0) {
        // Check remaining space to read
        const size_t dstSpaceLeft = dstSize - std::min(dstSize, mWholeBlocksSize);
        if (dstSpaceLeft == 0) {
            return true;
        }

        CFileGuard fileGuard(kFilePath, "rb");

        if (!fileGuard.status()) {
            LOG(ERROR) << "Error opening file.";
            return false;
        }
        auto file = fileGuard.file();

        const auto offset = mWholeBlocksSize;
        if (std::fseek(file, offset, SEEK_SET) != 0) {
            LOG(ERROR) << "Error seeking file.";
            return false;
        }
        const auto readSize = std::min(dstSpaceLeft, mLastBlockSize);
        const auto actualReadSize = std::fread((char*)dstBuffer + offset, 1, readSize, file);
        if (actualReadSize < readSize) {
            if (std::feof(file)) {
                LOG(DEBUG) << "End of file reached after reading " << actualReadSize << " bytes.";
            } else {
                LOG(ERROR) << "Error reading from file.";
                return false;
            }
        }

        const auto tailElapsed = timer.reset<Timer::ms>();
        const auto totalElapsed = asyncElapsed + tailElapsed;
        LOG(DEBUG) << "FileDirectReader read " << (mWholeBlocksSize + actualReadSize)
                   << " bytes, last remainder block size is " << mLastBlockSize << " bytes. "
                   << "Time taken: " << totalElapsed << " ms (async = " << asyncElapsed
                   << " ms, tail = " << tailElapsed << " ms).";
    } else {
        LOG(DEBUG) << "FileDirectReader read " << mWholeBlocksSize << " bytes."
                   << "Time taken: " << asyncElapsed << " ms.";
    }

    return true;
}
