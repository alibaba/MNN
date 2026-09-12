#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>

void MNNUnpackConvScaleFromBuffer_RVV(float* scaleBuffer, const int8_t* srcbuffer, const int32_t* info, int infoBytes) {
    const int blockNum = info[0];
    const int ocDiv = info[1];
    const int stride1 = info[2];
    const int unit = info[3];
    const size_t copyBytes = static_cast<size_t>(unit) * infoBytes;
    const size_t packedUnitSize = static_cast<size_t>(stride1) + 2 * copyBytes;
    int8_t* scaleWritePtr = reinterpret_cast<int8_t*>(scaleBuffer);

    for (int hU = 0; hU < ocDiv; ++hU) {
        const int8_t* huPtr = srcbuffer + static_cast<size_t>(hU) * blockNum * packedUnitSize;
        for (int bl = 0; bl < blockNum; ++bl) {
            const int8_t* scaleReadPtr = huPtr + static_cast<size_t>(bl) * packedUnitSize + stride1;
            size_t offset = 0;
            while (offset < copyBytes) {
                const size_t vl = __riscv_vsetvl_e8m8(copyBytes - offset);
                const vuint8m8_t data =
                    __riscv_vle8_v_u8m8(reinterpret_cast<const uint8_t*>(scaleReadPtr + offset), vl);
                __riscv_vse8_v_u8m8(reinterpret_cast<uint8_t*>(scaleWritePtr + offset), data, vl);
                offset += vl;
            }
            scaleWritePtr += copyBytes;
        }
    }
}
