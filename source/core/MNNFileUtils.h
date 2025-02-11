//
//  MNNFileUtils.h
//  MNN
//
//  Created by MNN on 2024/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_FileUtils_H
#define MNN_FileUtils_H

#include <stdio.h>
#include <stdint.h>
#include <string>
#include "core/Macro.h"
#include "MNN/ErrorCode.hpp"
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
#include <windows.h>
#include <io.h>
#undef max
#undef min
#undef NO_ERROR
#else
#include <unistd.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#endif

using namespace MNN;

#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    typedef HANDLE file_t;
    const file_t INVALID_FILE = INVALID_HANDLE_VALUE;
#else
    typedef int file_t;
    const file_t INVALID_FILE = -1;
#endif

#define MNN_FILE_READ   1U
#define MNN_FILE_WRITE  2U
#define INVALID_SIZE    ((size_t)(-1))

/*=============================================================================================
**  @brief      Concat a file name with a directory path
**  @hint       This function can be called multiple times to concat multi-level paths
*/
MNN_PUBLIC std::string MNNFilePathConcat(std::string prefix, std::string suffix);

/*=============================================================================================
**  @brief      Check whether a directory exists
**  @param      path -- path of the directory
**  @return     If the directory exists, returns true
**              If the directory does not exist, return false
*/
MNN_PUBLIC bool MNNDirExist(const char * path);

/*=============================================================================================
**  @brief      Create a directory if not exists
**  @param      path -- path of the directory
**  @return     If the directory exists or create success, returns true
**              If the directory does not exist and create fail, return false
*/
MNN_PUBLIC bool MNNCreateDir(const char * path);

/*=============================================================================================
**  @brief      Check whether a file exists
**  @param      file_name -- path of the file
**  @return     If the file exists, returns true
**              If the file does not exist, return false
*/
MNN_PUBLIC bool MNNFileExist(const char * file_name);

/*=============================================================================================
**  @brief      Create a file
**  @param      file_name -- path of the file
**  @return     If succeeded, returns the handle of the created file in the read and write mode
**              If failed, returns INVALID_FILE
**  @warning    If the file exists already, it will be covered
**              Size of the newly created file will be 0
*/
MNN_PUBLIC file_t MNNCreateFile(const char * file_name);

/*=============================================================================================
**  @brief      Open a file
**  @param      file_name -- path of the file
**              flags -- openning mode (MNN_FILE_READ or MNN_FILE_WRITE or both)
**  @return     If succeeded, returns the handle of the file
**              If failed, returns INVALID_FILE
**  @warning    If the file does not exist, this function would fail
**              Make sure that the aimed file has been created by MNNCreateFile()
*/
MNN_PUBLIC file_t MNNOpenFile(const char * file_name, uint32_t flags);

/*=============================================================================================
**  @brief      Close a file
**  @param      file -- handle of the file
**  @return     If succeeded, returns NO_ERROR
**              If failed, returns FAILED
**  @warning    Close an INVALID_FILE would fail
**              Make sure that the aimed file has been opened by MNNOpenFile()
*/
MNN_PUBLIC ErrorCode MNNCloseFile(file_t file);

/*=============================================================================================
**  @brief      Remove a file
**  @param      file_name -- path of the file
**  @return     If succeeded, returns NO_ERROR
**              If failed, returns FAILED
**  @warning    If the file does not exist, this function would fail
*/
MNN_PUBLIC ErrorCode MNNRemoveFile(const char * file_name);

/*=============================================================================================
**  @brief      Get the size of a file
**  @param      file -- handle of the file
**  @return     size of the file or INVALID_SIZE for INVALID_FILE
*/
MNN_PUBLIC size_t MNNGetFileSize(file_t file);

/*=============================================================================================
**  @brief      Resize a file
**  @param      file -- handle of the file
**              aimed_size -- the aimed size of this file
**  @return     If succeeded, returns NO_ERROR
**              If failed, returns FAILED
**  @warning    resize an INVALID_FILE would fail
*/
MNN_PUBLIC ErrorCode MNNSetFileSize(file_t file, size_t aimed_size);

/*=============================================================================================
**  @brief      Read from the file to the buffer
**  @param      file  -- handle of the file
**              buf   -- start address of the buffer in memory
**              bytes -- number of bytes to be read
**  @return     how many bytes have been read actually
**  @warning    Make sure that space of the buffer is enough
**              Otherwise, this function may access out of bounds
*/
MNN_PUBLIC size_t MNNReadFile(file_t file, void * buf, size_t bytes);

/*=============================================================================================
**  @brief      Write to the file from the buffer
**  @param      file  -- handle of the file
**              buf   -- start address of the buffer in memory
**              bytes -- number of bytes to be written
**  @return     how many bytes have been written actually
**  @warning    Make sure the data in the buffer is enough
**              Otherwise, this function may access out of bounds
*/
MNN_PUBLIC size_t MNNWriteFile(file_t file, void * buf, size_t bytes);

/*=============================================================================================
**  @brief      Set the file pointer to a given position
**  @param      file   -- handle of the file
**              offset -- the aimed postion from the start of the file
**  @return     If succeeded, returns NO_ERROR
**              If failed, returns FAILED
**  @warning    Make sure the offset not exceeding the file size
*/
MNN_PUBLIC ErrorCode MNNSetFilePointer(file_t file, size_t offset);

/*=============================================================================================
**  @brief      Memory-map the file to the virtual address space of the current process
**  @param      file -- handle of the file
**              size -- mapped length
**  @return     If succeeded, returns the start address of the mapped space
**              If failed, return nullptr
**  @hint       Memory-mapping a file to the virtual address space enables the process to access it by pointers
**              After the memory-mapping, the user can simply treat the mapped space as a memory buffer 
**              Read from or write to the mapped space will trigger data swapping
**              between the file on disk and the kernel page cache in memory
**              which is managed by the OS kernel and is transparent to the user
**  @warning    Make sure that the mapped size is no larger than the size of the file
**              Especially when mapping a newly created file, whose size is 0
*/
MNN_PUBLIC void * MNNMmapFile(file_t file, size_t size);

/*=============================================================================================
**  @brief      Unmap a previously mapped memory space
**  @param      addr -- start address of the mapped space
**              size -- mapped length
**  @return     If succeeded, returns NO_ERROR
**              If failed, returns FAILED
**  @warning    Make sure that this space was mapped by the MNNMmapFile() before
**              and the size is correct
*/
MNN_PUBLIC ErrorCode MNNUnmapFile(void * addr, size_t size);

/*=============================================================================================
**  @brief      Sync mmap memory to disk.
**  @param      addr -- start address of the mapped space
**              size -- mapped length
**  @return     If succeeded, returns NO_ERROR
**              If failed, returns FAILED
**  @warning    Make sure that this space was mapped by the MNNMmapFile() before
**              and the size is correct
*/
MNN_PUBLIC ErrorCode MNNMmapSync(void * addr, size_t size);

#endif // MNN_FileUtils_H
