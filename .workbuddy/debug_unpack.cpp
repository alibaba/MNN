#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

void MNNUnpackConvScaleFromBuffer_RVV(float* scaleBuffer, const int8_t* srcbuffer, const int32_t* info, int infoBytes);

int main() {
    for (int unit : {1, 4, 13}) {
        const int blockNum = 1, ocDiv = 1, stride1 = 64, infoBytes = 4;
        const int32_t info[4] = {blockNum, ocDiv, stride1, unit};
        const size_t copyBytes = static_cast<size_t>(unit) * infoBytes;
        const size_t packedUnitSize = static_cast<size_t>(stride1) + 2 * copyBytes;
        std::vector<int8_t> src(ocDiv * blockNum * packedUnitSize, 0);
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = static_cast<int8_t>((i * 37 + 11) % 251 - 125);
        std::vector<float> rvv(ocDiv * blockNum * unit), ref(ocDiv * blockNum * unit);
        MNNUnpackConvScaleFromBuffer_RVV(rvv.data(), src.data(), info, infoBytes);
        int8_t* w = reinterpret_cast<int8_t*>(ref.data());
        for (int hU = 0; hU < ocDiv; ++hU)
            for (int bl = 0; bl < blockNum; ++bl) {
                const int8_t* r = src.data() + (size_t)hU * blockNum * packedUnitSize + (size_t)bl * packedUnitSize + stride1;
                memcpy(w, r, copyBytes);
                w += copyBytes;
            }
        size_t bad = SIZE_MAX, count = 0;
        for (size_t i = 0; i < ref.size() * 4; ++i) {
            if (((int8_t*)rvv.data())[i] != ((int8_t*)ref.data())[i]) {
                if (bad == SIZE_MAX) bad = i;
                count++;
            }
        }
        printf("unit=%d copyBytes=%zu mismatches=%zu first_bad=%zu", unit, copyBytes, count, bad);
        if (bad != SIZE_MAX) {
            printf(" | rvv=0x%02x ref=0x%02x src[64+bad-1..+2]=0x%02x,0x%02x,0x%02x",
                   (unsigned char)((int8_t*)rvv.data())[bad], (unsigned char)((int8_t*)ref.data())[bad],
                   (unsigned char)src[64 + bad - 1], (unsigned char)src[64 + bad], (unsigned char)src[64 + bad + 1]);
        }
        printf("\n");
    }
    return 0;
}
