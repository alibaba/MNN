//
//  MetalShaderCodec.hpp
//  MNN
//
//  Decoder for build-time compressed Metal shader source.
//

#ifndef MetalShaderCodec_hpp
#define MetalShaderCodec_hpp

#include <stddef.h>
#include <stdint.h>

namespace MNN {

// One build-time compressed Metal shader source blob.
struct MetalShaderBlob {
    const uint8_t* data;
    uint32_t packedSize;
    uint32_t rawSize;
};

// Decodes blob into a NUL-terminated buffer of blob.rawSize + 1 bytes.
// Returns false and leaves dst untouched when the stream is malformed.
bool MetalShaderDecode(const MetalShaderBlob& blob, char* dst);

// Returns the decoded source for blob, caching it for the process lifetime.
// Thread-safe. Never returns nullptr: an empty string is returned when the
// stream is malformed or allocation fails, which surfaces as a normal Metal
// compile failure at the call site.
const char* MetalShaderGet(const MetalShaderBlob& blob);

} // namespace MNN

#endif /* MetalShaderCodec_hpp */
