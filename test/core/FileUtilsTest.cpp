//
//  FileUtilsTest.cpp
//  MNNTests
//
//  Created by MNN on 2024/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNTestSuite.h"
#include "core/MNNFileUtils.h"

#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
const char * file_path = "C:\\Windows\\Temp\\file_utils_test_temp_file";
#elif defined(__ANDROID__)
const char * file_path = "/data/local/tmp/file_utils_test_temp_file";
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix__)
const char * file_path = "/tmp/file_utils_test_temp_file";
#else
const char * file_path = "./file_utils_test_temp_file";
#endif

class FileUtilsTest : public MNNTestCase {
public:
    virtual ~FileUtilsTest() = default;
    virtual bool run(int precision) {
        /*======== Create and Remove ========*/
        do {
            // create a new file
            file_t file = MNNCreateFile(file_path);
            if (file == INVALID_FILE) {
                return false;
            } else {
                MNNCloseFile(file);
            }
            bool exist = MNNFileExist(file_path);
            if (!exist) {
                return false;
            }
            // create a new file to cover an existing file
            file = MNNCreateFile(file_path);
            if (file == INVALID_FILE) {
                return false;
            } else {
                MNNCloseFile(file);
            }
            exist = MNNFileExist(file_path);
            if (!exist) {
                return false;
            }
            // remove a file
            MNN::ErrorCode ec = MNNRemoveFile(file_path);
            if (ec != NO_ERROR) {
                return false;
            }
            exist = MNNFileExist(file_path);
            if (exist) {
                return false;
            }
            printf("File Utils Test: Create and Remove passed\n");
        } while(false);

        /*======== Open and Close ========*/
        do {
            // Open and close a non-existent file
            file_t file = MNNOpenFile(file_path, MNN_FILE_READ | MNN_FILE_WRITE);
            if (file != INVALID_FILE) {
                return false;
            }
            MNN::ErrorCode ec = MNNCloseFile(file);
            if (ec != FILE_NOT_EXIST) {
                return false;
            }
            // Open and close an existent file
            file = MNNCreateFile(file_path);
            if (file == INVALID_FILE) {
                return false;
            }
            bool exist = MNNFileExist(file_path);
            if (!exist) {
                return false;
            }
            ec = MNNCloseFile(file);
            if (ec != NO_ERROR) {
                return false;
            }
            file = MNNOpenFile(file_path, MNN_FILE_READ | MNN_FILE_WRITE);
            if (file == INVALID_FILE) {
                return false;
            }
            ec = MNNCloseFile(file);
            if (ec != NO_ERROR) {
                return false;
            }
            ec = MNNRemoveFile(file_path);
            if (ec != NO_ERROR) {
                return false;
            }
            exist = MNNFileExist(file_path);
            if (exist) {
                return false;
            }
            printf("File Utils Test: Open and Close passed\n");
        } while(false);

        /*======== Get and Set File Size ========*/
        do {
            file_t file = MNNOpenFile(file_path, MNN_FILE_READ | MNN_FILE_WRITE);
            if (file != INVALID_FILE) {
                return false;
            }
            size_t size = MNNGetFileSize(file);
            if (size != INVALID_SIZE) {
                printf("File size mismatch: expected %lu but got %lu\n", INVALID_SIZE, size);
                return false;
            }
            file = MNNCreateFile(file_path);
            if (file == INVALID_FILE) {
                return false;
            }
            size = MNNGetFileSize(file);
            if (size != 0) {
                printf("File size mismatch: expected 0 but got %lu\n", size);
                return false;
            }
            size_t expectedSize = 1023;
            MNN::ErrorCode ec = MNNSetFileSize(file, expectedSize);
            if (ec != NO_ERROR) {
                return false;
            }
            size = MNNGetFileSize(file);
            if (size != expectedSize) {
                printf("File size mismatch: expected %lu but got %lu\n", expectedSize, size);
                return false;
            }
            expectedSize = 64 * 1024 * 1024;
            ec = MNNSetFileSize(file, expectedSize);
            if (ec != NO_ERROR) {
                return false;
            }
            size = MNNGetFileSize(file);
            if (size != expectedSize) {
                printf("File size mismatch: expected %lu but got %lu\n", expectedSize, size);
                return false;
            }
            ec = MNNCloseFile(file);
            if (ec != NO_ERROR) {
                return false;
            }
            ec = MNNRemoveFile(file_path);
            if (ec != NO_ERROR) {
                return false;
            }
            bool exist = MNNFileExist(file_path);
            if (exist) {
                return false;
            }
            printf("File Utils Test: Get and Set File Size passed\n");
        } while(false);

        /*======== Read and Write ========*/
        do {
            char alpha[27] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
            size_t size = 32;
            char * buf = (char *)malloc(size);
            if (buf == nullptr) {
                printf("MNN_FAILED to allocate memory in File Utils Test!\n");
                return false;
            }
            file_t file = MNNOpenFile(file_path, MNN_FILE_READ | MNN_FILE_WRITE);
            if (file != INVALID_FILE) {
                return false;
            }
            size_t ret = MNNReadFile(file, nullptr, 0);
            if (ret != 0) {
                return false;
            }
            ret = MNNWriteFile(file, nullptr, 0);
            if (ret != 0) {
                return false;
            }
            file = MNNCreateFile(file_path);
            if (file == INVALID_FILE) {
                return false;
            }
            ret = MNNReadFile(file, buf, 10);
            if (ret != 0) {
                return false;
            }
            ret = MNNWriteFile(file, alpha, 26);
            if (ret != 26) {
                return false;
            }
            MNN::ErrorCode ec = MNNSetFilePointer(file, 0);
            if (ec != NO_ERROR) {
                return false;
            }
            ret = MNNReadFile(file, buf, 20);
            if (ret != 20) {
                return false;
            }
            ret = MNNReadFile(file, buf, 10);
            if (ret != 6) {
                return false;
            }
            ec = MNNSetFilePointer(file, 0);
            if (ec != NO_ERROR) {
                return false;
            }
            ret = MNNReadFile(file, buf, 3);
            if (ret != 3) {
                return false;
            }
            if (buf[0] != 'A' || buf[1] != 'B' || buf[2] != 'C') {
                return false;
            }
            ec = MNNSetFilePointer(file, 20);
            if (ec != NO_ERROR) {
                return false;
            }
            ret = MNNReadFile(file, buf, 3);
            if (ret != 3) {
                return false;
            }
            if (buf[0] != 'U' || buf[1] != 'V' || buf[2] != 'W') {
                return false;
            }
            ec = MNNSetFilePointer(file, 10);
            if (ec != NO_ERROR) {
                return false;
            }
            char hello[6] = "hello";
            ret = MNNWriteFile(file, (void *)hello, 6);
            if (ret != 6) {
                return false;
            }
            ec = MNNSetFilePointer(file, 10);
            if (ec != NO_ERROR) {
                return false;
            }
            ret = MNNReadFile(file, buf, 6);
            if (0 != strcmp(buf, "hello")) {
                return false;
            }
            ec = MNNCloseFile(file);
            if (ec != NO_ERROR) {
                return false;
            }
            ec = MNNRemoveFile(file_path);
            if (ec != NO_ERROR) {
                return false;
            }
            bool exist = MNNFileExist(file_path);
            if (exist) {
                return false;
            }
            free(buf);
            printf("File Utils Test: Read and Write passed\n");
        } while(false);

        /*======== Map and Unmap ========*/
        do {
            char * addr = (char *)MNNMmapFile(INVALID_FILE, INVALID_SIZE);
            if (addr != nullptr) {
                return false;
            }
            MNN::ErrorCode ec = MNNUnmapFile(addr, 0);
            if (ec != FILE_UNMAP_FAILED) {
                return false;
            }
            file_t file = MNNCreateFile(file_path);
            if (file == INVALID_FILE) {
                return false;
            }
            addr = (char *)MNNMmapFile(file, 1024);
            if (addr != nullptr) {
                return false;
            }
            ec = MNNSetFileSize(file, 1024);
            if (ec != NO_ERROR) {
                return false;
            }
            addr = (char *)MNNMmapFile(file, 1024);
            if (addr == nullptr) {
                return false;
            }
            strcpy(addr, "hello");
            ec = MNNUnmapFile(addr, 1024);
            if (ec != NO_ERROR) {
                return false;
            }
            addr = (char *)MNNMmapFile(file, 1024);
            if (addr == nullptr) {
                return false;
            }
            if(0 != strcmp(addr, "hello")) {
                return false;
            }
            ec = MNNUnmapFile(addr, 1024);
            if (ec != NO_ERROR) {
                return false;
            }
            ec = MNNCloseFile(file);
            if (ec != NO_ERROR) {
                return false;
            }
            ec = MNNRemoveFile(file_path);
            if (ec != NO_ERROR) {
                return false;
            }
            bool exist = MNNFileExist(file_path);
            if (exist) {
                return false;
            }
            printf("File Utils Test: Map and Unmap passed\n");
        } while(false);

        return true;
    }
};
MNNTestSuiteRegister(FileUtilsTest, "core/file_utils");
