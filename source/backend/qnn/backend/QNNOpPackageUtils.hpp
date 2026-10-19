//
//  QNNOpPackageUtils.hpp
//  MNN
//

#ifndef MNN_QNN_OP_PACKAGE_UTILS_HPP
#define MNN_QNN_OP_PACKAGE_UTILS_HPP

#include <cstddef>
#include <string>

namespace MNN {
namespace QNN {

inline bool deriveV66LayerNormOpPackageNames(const std::string& libraryName, std::string* interfaceProvider,
                                             std::string* packageName) {
    if (interfaceProvider == nullptr || packageName == nullptr) {
        return false;
    }
    constexpr char kPrefix[] = "lib";
    constexpr char kSuffix[] = "V66LayerNorm.so";
    constexpr size_t kPrefixLength = sizeof(kPrefix) - 1;
    constexpr size_t kSuffixLength = sizeof(kSuffix) - 1;
    if (libraryName.size() <= kPrefixLength + kSuffixLength || libraryName.compare(0, kPrefixLength, kPrefix) != 0 ||
        libraryName.compare(libraryName.size() - kSuffixLength, kSuffixLength, kSuffix) != 0) {
        return false;
    }
    const size_t nameLength = libraryName.size() - kPrefixLength - kSuffixLength;
    const char first = libraryName[kPrefixLength];
    const bool firstIsLetter = (first >= 'a' && first <= 'z') || (first >= 'A' && first <= 'Z');
    if (!firstIsLetter) {
        return false;
    }
    for (size_t index = 1; index < nameLength; ++index) {
        const char character = libraryName[kPrefixLength + index];
        const bool valid = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
                           (character >= '0' && character <= '9') || character == '_';
        if (!valid) {
            return false;
        }
    }
    const std::string stem = libraryName.substr(kPrefixLength, nameLength);
    *interfaceProvider = stem + "V66LayerNormPackageInterfaceProvider";
    *packageName = stem + "V66LayerNormPackage";
    return true;
}

} // namespace QNN
} // namespace MNN

#endif // MNN_QNN_OP_PACKAGE_UTILS_HPP
