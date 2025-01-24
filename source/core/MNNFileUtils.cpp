//
//  MNNFileUtils.cpp
//  MNN
//
//  Created by MNN on 2024/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cstring>
#include "MNNFileUtils.h"

std::string MNNFilePathConcat(std::string prefix, std::string suffix) {
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    return prefix + "\\" + suffix;
#else
    return prefix + "/" + suffix;
#endif
}

bool MNNDirExist(const char * path) {
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    DWORD file_attributes = GetFileAttributes(path);
    return (file_attributes != INVALID_FILE_ATTRIBUTES) && (file_attributes & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat info;
    return (stat(path, &info) == 0) && (info.st_mode & S_IFDIR);
#endif
}

bool MNNFileExist(const char * file_name)
{
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    return _access(file_name, 0) == 0;
#else
    return access(file_name, F_OK) == 0;
#endif
}

bool MNNCreateDir(const char * path) {
    if (MNNDirExist(path)) {
        return true;
    }
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    if (CreateDirectory(path, NULL) || ERROR_ALREADY_EXISTS == GetLastError()) {
        return true;
    } else {
        return false;
    }
#else
    if (mkdir(path, 0755) == 0) {
        return true;
    }
    return MNNDirExist(path);
#endif
}

file_t MNNCreateFile(const char * file_name)
{
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    HANDLE hd = CreateFile(
        file_name,                      // File Name
        GENERIC_READ | GENERIC_WRITE,   // Read and Write
        0,                              // No Sharing
        NULL,                           // No Security
        CREATE_ALWAYS,                  // Create the file and cover the existing file
        FILE_ATTRIBUTE_NORMAL,          // Normal Attribute
        NULL                            // No Template
    );
    if (hd == INVALID_HANDLE_VALUE) {
        MNN_PRINT("Failed to create the file: %s\n", file_name);
        return INVALID_FILE;
    }
    return hd;
#else
    int fd = open(
        file_name,                      // File Name
        O_RDWR | O_CREAT | O_TRUNC,     // Read and Write and Create the file and cover existing file
        0666                            // Read and Write Permission for Everyone
    );
    if (fd == -1) {
        MNN_PRINT("Failed to create the file: %s\n", file_name);
        return INVALID_FILE;
    }
    return fd;
#endif
}

file_t MNNOpenFile(const char * file_name, uint32_t flags)
{
    if (!MNNFileExist(file_name)) {
        MNN_PRINT("File not exist: %s\n", file_name);
        return INVALID_FILE;
    }
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    DWORD mode = 0;
    if (flags & MNN_FILE_READ) {
        mode |= GENERIC_READ;
    }
    if (flags & MNN_FILE_WRITE) {
        mode |= GENERIC_WRITE;
    }
    HANDLE hd = CreateFile(
        file_name,              // File Name
        mode,                   // Opening Mode
        0,                      // No Sharing
        NULL,                   // No Security
        OPEN_EXISTING,          // Only Open Existing File
        FILE_ATTRIBUTE_NORMAL,  // Normal Attribute
        NULL                    // No Template
    );
    if (hd == INVALID_HANDLE_VALUE) {
        MNN_PRINT("Failed to open the file: %s\n", file_name);
        return INVALID_FILE;
    }
    return hd;
#else
    int mode = 0;
    if (flags & MNN_FILE_READ) {
        mode = O_RDONLY;
    }
    if (flags & MNN_FILE_WRITE) {
        mode = O_RDWR;
    }
    int fd = open(file_name, mode);
    if (fd == -1) {
        MNN_PRINT("Failed to open the file: %s\n", file_name);
        return INVALID_FILE;
    }
    return fd;
#endif
}

ErrorCode MNNCloseFile(file_t file)
{
    if (file == INVALID_FILE) {
        return FILE_NOT_EXIST;
    }
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    if (!CloseHandle(file)) {
        return FILE_CLOSE_FAILED;
    }
#else
    if (-1 == close(file)) {
        return FILE_CLOSE_FAILED;
    }
#endif
    return NO_ERROR;
}

ErrorCode MNNRemoveFile(const char * file_name)
{
    if (!MNNFileExist(file_name)) {
        return FILE_NOT_EXIST;
    }
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    if (!DeleteFile(file_name)) {
        return FILE_REMOVE_FAILED;
    }
#else
    if (-1 == unlink(file_name)) {
        return FILE_REMOVE_FAILED;
    }
#endif
    return NO_ERROR;
}

size_t MNNGetFileSize(file_t file)
{
    if (file == INVALID_FILE) {
        return INVALID_SIZE;
    }
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(file, &fileSize)) {
        return (size_t)(-1);
    } else {
        return (size_t)(fileSize.QuadPart);
    }
#else
    struct stat file_stat;
    if (fstat(file, &file_stat) == -1) {
        return (size_t)(-1);
    } else {
        return file_stat.st_size;
    }
#endif
}

ErrorCode MNNSetFileSize(file_t file, size_t aimed_size)
{
    if (file == INVALID_FILE) {
        return FILE_NOT_EXIST;
    }
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    LARGE_INTEGER size;
    size.QuadPart = aimed_size;
    bool success = SetFilePointerEx(file, size, NULL, FILE_BEGIN);
    if (!success) {
        return FILE_RESIZE_FAILED;
    }
    success = SetEndOfFile(file);
    if (!success) {
        return FILE_RESIZE_FAILED;
    }
    return NO_ERROR;
#else
    if (-1 == ftruncate(file, aimed_size)) {
        return FILE_RESIZE_FAILED;
    }
    return NO_ERROR;
#endif
}

size_t MNNReadFile(file_t file, void * buf, size_t bytes)
{
    if (file == INVALID_FILE || buf == nullptr) {
        return 0;
    }
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    DWORD readbytes = 0;
    if (ReadFile(file, buf, bytes, &readbytes, NULL)) {
        return readbytes;
    } else {
        return 0;
    }
#else
    return read(file, buf, bytes);
#endif
}

size_t MNNWriteFile(file_t file, void * buf, size_t bytes)
{
    if (file == INVALID_FILE || buf == nullptr) {
        return 0;
    }
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    DWORD writebytes = 0;
    if (WriteFile(file, buf, bytes, &writebytes, NULL)) {
        return writebytes;
    } else {
        return 0;
    }
#else
    return write(file, buf, bytes);
#endif
}

ErrorCode MNNSetFilePointer(file_t file, size_t offset)
{
    if (file == INVALID_FILE) {
        return FILE_NOT_EXIST;
    }
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    LARGE_INTEGER liDistanceToMove;
    liDistanceToMove.QuadPart = offset;
    if (SetFilePointerEx(file, liDistanceToMove, NULL, FILE_BEGIN)) {
        return NO_ERROR;
    } else {
        return FILE_SEEK_FAILED;
    }
#else
    if (-1 == lseek(file, offset, SEEK_SET)) {
        return FILE_SEEK_FAILED;
    } else {
        return NO_ERROR;
    }
#endif
}

void * MNNMmapFile(file_t file, size_t size)
{
    if (file == INVALID_FILE || MNNGetFileSize(file) < size) {
        return nullptr;
    }
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    HANDLE hFileMapping = CreateFileMapping(file, NULL, PAGE_READWRITE, (size >> 32) & 0xffffffff, size & 0xffffffff, NULL);
    if (hFileMapping == NULL) {
        MNN_ERROR("MNN: Mmap failed\n");
        return nullptr;
    }
    void * addr = MapViewOfFile(hFileMapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, size);
    CloseHandle(hFileMapping);
    return addr;
#else
    void * addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0);
    if (addr == MAP_FAILED) {
        MNN_ERROR("MNN: Mmap failed\n");
        return nullptr;
    }
    return addr;
#endif
}

ErrorCode MNNUnmapFile(void * addr, size_t size)
{
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    if (!UnmapViewOfFile(addr)) {
        return FILE_UNMAP_FAILED;
    }
#else
    if (-1 == munmap(addr, size)) {
        return FILE_UNMAP_FAILED;
    }
#endif
    return NO_ERROR;
}

ErrorCode MNNMmapSync(void * addr, size_t size)
{
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    if (!FlushViewOfFile(addr, 0)) {
        return FILE_UNMAP_FAILED;
    }
#else
    if (-1 == msync(addr, size, MS_SYNC)) {
        return FILE_UNMAP_FAILED;
    }
#endif
    return NO_ERROR;
}
