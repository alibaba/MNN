//
//  MetalShaderCodec.cpp
//  MNN
//
//  Decoder for build-time compressed Metal shader source.
//
//  Stream format produced by source/backend/metal/packshader.py. Byte-oriented
//  LZ77 with a repeat-offset slot; no entropy coding, so the decoder stays
//  small and needs no external dependency.
//
//    literal token   0LLLLLLL  L in 1..127 raw bytes follow
//                    00000000  varint length, then that many raw bytes
//    match token     10LLLLLL  L = len-4 in 0..62, then varint offset
//                    10111111  varint len-4, then varint offset
//                    11LLLLLL  same lengths, reuses the previous offset
//
//  Matches may overlap the current output position, so copying must be
//  byte-by-byte forwards (run-length semantics).
//

#include "MetalShaderCodec.hpp"
#include <mutex>
#include <string>
#include <unordered_map>

namespace MNN {

bool MetalShaderDecode(const MetalShaderBlob& blob, char* dst) {
    const uint8_t* src = blob.data;
    const uint8_t* end = src + blob.packedSize;
    const uint32_t rawSize = blob.rawSize;
    uint32_t out = 0;
    uint32_t rep = 0;

    // Reads a LEB128 length/offset. Bounded to 5 groups so a corrupt stream
    // cannot spin, and rejects values that cannot address this blob.
    auto readVarint = [&](uint32_t& value) -> bool {
        uint32_t result = 0;
        for (int shift = 0; shift < 35; shift += 7) {
            if (src >= end) {
                return false;
            }
            uint8_t byte = *src++;
            result |= (uint32_t)(byte & 0x7f) << shift;
            if (0 == (byte & 0x80)) {
                value = result;
                return true;
            }
        }
        return false;
    };

    while (out < rawSize) {
        if (src >= end) {
            return false;
        }
        uint8_t ctrl = *src++;
        if (0 == (ctrl & 0x80)) {
            uint32_t length = ctrl;
            if (0 == length && !readVarint(length)) {
                return false;
            }
            if (length > rawSize - out || (uint32_t)(end - src) < length) {
                return false;
            }
            for (uint32_t i = 0; i < length; ++i) {
                dst[out++] = (char)*src++;
            }
            continue;
        }
        uint32_t extra = ctrl & 0x3f;
        if (0x3f == extra && !readVarint(extra)) {
            return false;
        }
        uint32_t length = extra + 4;
        uint32_t offset = rep;
        if (0 == (ctrl & 0x40)) {
            if (!readVarint(offset)) {
                return false;
            }
            rep = offset;
        }
        if (0 == offset || offset > out || length > rawSize - out) {
            return false;
        }
        uint32_t from = out - offset;
        for (uint32_t i = 0; i < length; ++i) {
            dst[out++] = dst[from++];
        }
    }
    if (src != end) {
        return false;
    }
    dst[rawSize] = '\0';
    return true;
}

const char* MetalShaderGet(const MetalShaderBlob& blob) {
    // Keyed on the blob address: every shader has exactly one static blob, and
    // callers hold the returned pointer indefinitely, so entries are never
    // evicted. Function-local statics keep this out of the module initializer.
    static std::mutex gMutex;
    static std::unordered_map<const uint8_t*, std::string> gCache;

    std::lock_guard<std::mutex> guard(gMutex);
    auto iter = gCache.find(blob.data);
    if (iter != gCache.end()) {
        return iter->second.c_str();
    }
    std::string decoded;
    // The decoder terminates at dst[rawSize], so the buffer needs one extra
    // writable byte; shrink afterwards so size() excludes the terminator.
    decoded.resize((size_t)blob.rawSize + 1);
    if (MetalShaderDecode(blob, &decoded[0])) {
        decoded.resize(blob.rawSize);
    } else {
        decoded.clear();
    }
    auto result = gCache.emplace(blob.data, std::move(decoded));
    return result.first->second.c_str();
}

} // namespace MNN
