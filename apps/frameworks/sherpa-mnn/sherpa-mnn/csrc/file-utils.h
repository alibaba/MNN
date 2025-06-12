// sherpa-mnn/csrc/file-utils.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FILE_UTILS_H_
#define SHERPA_ONNX_CSRC_FILE_UTILS_H_

#include <fstream>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

namespace sherpa_mnn {

/** Check whether a given path is a file or not
 *
 * @param filename Path to check.
 * @return Return true if the given path is a file; return false otherwise.
 */
bool FileExists(const std::string &filename);

/** Abort if the file does not exist.
 *
 * @param filename The file to check.
 */
void AssertFileExists(const std::string &filename);

std::vector<char> ReadFile(const std::string &filename);

#if __ANDROID_API__ >= 9
std::vector<char> ReadFile(AAssetManager *mgr, const std::string &filename);
#endif

#if __OHOS__
std::vector<char> ReadFile(NativeResourceManager *mgr,
                           const std::string &filename);
#endif

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_FILE_UTILS_H_
