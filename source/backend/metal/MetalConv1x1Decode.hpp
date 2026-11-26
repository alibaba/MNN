#ifndef MetalConv1x1Decode_hpp
#define MetalConv1x1Decode_hpp

#include <algorithm>
#include <cstddef>

namespace MNN {

constexpr int kMetalSimdGroupWidth = 32;
constexpr int kMetalCacheLineBytes = 128;

// GPU device class, not the final decode profile. conv1x1ProfileFrom() combines
// this class with the GPU generation; Unknown gets a conservative profile with
// no threadgroup budget. Max/Ultra rows copy Pro's numbers until measured.
enum class Conv1x1DieClass {
    Base,
    Pro,
    Max,
    Ultra,
    Unknown
};

inline bool conv1x1AtLeastPro(Conv1x1DieClass die) {
    return die == Conv1x1DieClass::Pro || die == Conv1x1DieClass::Max ||
           die == Conv1x1DieClass::Ultra;
}

struct Conv1x1GpuTier {
    Conv1x1DieClass dieClass = Conv1x1DieClass::Unknown;
    bool prefersOc16Sg8K2;
    bool hasOc8Sg4K2Budget;
    bool highBandwidth;
    bool belowGen16;
    bool decodeFp32Validated;
};

enum class Conv1x1ProfileSource {
    Exact,
    NearestGen,
    UnknownGen,
    UnknownTier,
};

struct Conv1x1DeviceProfile {
    const char* name;
    Conv1x1GpuTier tier;
    int tgBudget;
    int gateUpSkQuadsFloor;
    bool gateUpSkDeviceClass;
    Conv1x1ProfileSource source;
};

struct Conv1x1ProfileEntry {
    const char* name;
    bool prefersOc16Sg8K2;
    bool hasOc8Sg4K2Budget;
    bool belowGen16;
    bool decodeFp32Validated;
    int tgBudget;
    int gateUpSkQuadsFloor;
};

enum Conv1x1ProfileRowId {
    Conv1x1RowPro = 0,
    Conv1x1RowMax,
    Conv1x1RowUltra,
    Conv1x1RowBase16,
    Conv1x1RowBase17,
    Conv1x1RowNoBudget32,
    Conv1x1RowNoBudget4,
};

inline const Conv1x1ProfileEntry& conv1x1ProfileRow(int idx) {
    static const Conv1x1ProfileEntry kConv1x1Profiles[] = {
        {"pro160",     false, false, false, true,  160, 32},
        {"max160",     false, false, false, true,  160, 32},
        {"ultra160",   false, false, false, true,  160, 32},
        {"base16",     true,  true,  false, true,   80,  4},
        {"base17",     false, true,  false, true,   80,  4},
        {"nobudget32", false, false, true,  false,   0, 32},
        {"nobudget4",  false, false, false, false,   0,  4},
    };
    return kConv1x1Profiles[idx];
}

inline Conv1x1DeviceProfile conv1x1ProfileFrom(Conv1x1DieClass die, int gen) {
    const bool knownGen = gen >= 13; // M1
    int idx;
    Conv1x1ProfileSource source;
    if (conv1x1AtLeastPro(die)) {
        idx = die == Conv1x1DieClass::Pro ? Conv1x1RowPro
            : (die == Conv1x1DieClass::Max ? Conv1x1RowMax : Conv1x1RowUltra);
        source = gen == 0 ? Conv1x1ProfileSource::UnknownGen
                          : (knownGen ? Conv1x1ProfileSource::Exact : Conv1x1ProfileSource::NearestGen);
    } else if (die == Conv1x1DieClass::Base) {
        if (gen == 16) {
            idx = Conv1x1RowBase16;
            source = Conv1x1ProfileSource::Exact;
        } else if (gen >= 17) {
            idx = Conv1x1RowBase17;
            source = Conv1x1ProfileSource::Exact;
        } else {
            idx = Conv1x1RowNoBudget32;
            source = gen == 0 ? Conv1x1ProfileSource::UnknownGen
                              : (knownGen ? Conv1x1ProfileSource::Exact : Conv1x1ProfileSource::NearestGen);
        }
    } else {
        idx = gen >= 16 ? Conv1x1RowNoBudget4 : Conv1x1RowNoBudget32;
        source = Conv1x1ProfileSource::UnknownTier;
    }
    const Conv1x1ProfileEntry& e = conv1x1ProfileRow(idx);
    Conv1x1DeviceProfile p;
    p.name = e.name;
    p.tier.dieClass = die;
    p.tier.prefersOc16Sg8K2 = e.prefersOc16Sg8K2;
    p.tier.hasOc8Sg4K2Budget = e.hasOc8Sg4K2Budget;
    p.tier.highBandwidth = conv1x1AtLeastPro(die);
    p.tier.belowGen16 = e.belowGen16;
    p.tier.decodeFp32Validated = e.decodeFp32Validated;
    p.tgBudget = e.tgBudget;
    p.gateUpSkQuadsFloor = e.gateUpSkQuadsFloor;
    p.gateUpSkDeviceClass = e.hasOc8Sg4K2Budget;
    p.source = source;
    return p;
}

inline int decodeUpDiv(int a, int b) {
    return b > 0 ? (a + b - 1) / b : 0;
}

inline int gemvLanesPerBlock(int inputDepthQuad, int blockCount, bool splitK) {
    if (blockCount <= 0) {
        return 0;
    }
    const int quadsPerBlock = decodeUpDiv(inputDepthQuad, blockCount);
    const int legacyLanes    = std::min(kMetalSimdGroupWidth, std::max(quadsPerBlock / 4, 1));
    if (!splitK) {
        return legacyLanes;
    }
    const int blocksPerSimdgroup    = std::max(blockCount / 2, 1);
    const int oneBlockPerLaneGroup  = std::max(kMetalSimdGroupWidth / blocksPerSimdgroup, 1);
    const int pairLaneGroups        = kMetalSimdGroupWidth / 2;
    const bool shortSecondPass      = blocksPerSimdgroup <= pairLaneGroups + pairLaneGroups / 2;
    const int floorLanes            = shortSecondPass ? 1 : 2;
    const int chosenLanes =
        std::max(std::min(legacyLanes, oneBlockPerLaneGroup), std::min(legacyLanes, floorLanes));
    int pow2Lanes = 1;
    while (pow2Lanes * 2 <= chosenLanes) {
        pow2Lanes *= 2;
    }
    return pow2Lanes;
}

inline int gemvSplitKFillLanes(int blockCount) {
    const int blocksOwned = std::max(blockCount / 2, 1);
    int bestLanes = 2;
    int bestSlots = 0;
    for (int lanes = 2; lanes <= 4; lanes *= 2) {
        const int blocksInFlight = kMetalSimdGroupWidth / lanes;
        const int slots = decodeUpDiv(blocksOwned, blocksInFlight) * blocksInFlight;
        if (bestSlots == 0 || slots < bestSlots) {
            bestSlots = slots;
            bestLanes = lanes;
        }
    }
    return bestLanes;
}

inline int gemvQuadsPerCacheLine(int dequantBits) {
    const int quads = std::max(kMetalCacheLineBytes / std::max(2 * dequantBits, 1), 1);
    int pow2Quads = 1;
    while (pow2Quads * 2 <= quads) {
        pow2Quads *= 2;
    }
    return pow2Quads;
}

inline int gemvCacheLineLanes(int inputDepthQuad, int blockCount, int dequantBits, int lanes) {
    if (blockCount <= 0 || dequantBits <= 0) {
        return lanes;
    }
    const int quadsPerBlock = decodeUpDiv(inputDepthQuad, blockCount);
    const int lineQuads     = gemvQuadsPerCacheLine(dequantBits);
    if (quadsPerBlock <= lineQuads) {
        return lanes;
    }
    return std::min(kMetalSimdGroupWidth, std::max(lanes, lineQuads));
}

inline int gemvPlainLanesPerBlock(int inputDepthQuad, int blockCount, int dequantBits) {
    const int defaultLanes = gemvLanesPerBlock(inputDepthQuad, blockCount, /*splitK=*/false);
    if (defaultLanes <= 0) {
        return 0;
    }
    const int lanes = gemvCacheLineLanes(inputDepthQuad, blockCount, dequantBits, defaultLanes);
    return lanes == defaultLanes ? 0 : lanes;
}

inline bool oc16PreferredOverSplitK(int quadsPerBlock) {
    return quadsPerBlock >= 32 && quadsPerBlock < 96;
}

inline bool oc16EligibleVsPlain(int quadsPerBlock, bool belowGen16) {
    return quadsPerBlock >= 32 || belowGen16;
}

enum class DecodeGemvVariant { Oc8Sg4K2, Oc16Sg4K1, Plain };

struct DecodeGemvPlan {
    DecodeGemvVariant variant;
    int lanesPerBlock;
    int tgQuads;
};

struct DecodeCaps {
    Conv1x1GpuTier tier = {};
    int tgBudget = 0;
    bool wqEvenBlockFp16 = false;
};

struct DecodeShape {
    int oc = 0;
    int icDiv4 = 0;
    int blockCount = 1;
    int dequantBits = 0;
};

inline DecodeGemvPlan selectDecodeGemv(const DecodeCaps& caps, const DecodeShape& shape) {
    const int oc = shape.oc;
    const int icDiv4 = shape.icDiv4;
    const int blockCount = shape.blockCount;
    const int quadsPerBlock = decodeUpDiv(icDiv4, blockCount);
    const bool tgBudgetRoute = caps.tgBudget > 0 && caps.wqEvenBlockFp16 && quadsPerBlock >= 32 && oc % 16 == 0;
    const bool highBwTgBudgetRoute = tgBudgetRoute && caps.tier.highBandwidth;
    const int oc16Threadgroups = oc / 16;
    const bool splitKFitsBudget = tgBudgetRoute && 2 * oc16Threadgroups <= caps.tgBudget;
    const int oc16GridLimit = caps.tier.highBandwidth ? 2 * caps.tgBudget : caps.tgBudget;
    const bool oc16ExceedsLimit = tgBudgetRoute && oc16Threadgroups > oc16GridLimit;
    bool useOc8Sg4K2 = oc % 8 == 0 && blockCount % 2 == 0 &&
        (splitKFitsBudget || oc16ExceedsLimit || (!oc16PreferredOverSplitK(quadsPerBlock) && !highBwTgBudgetRoute));
    if (caps.tier.belowGen16 && oc % 16 == 0 && quadsPerBlock <= 96) {
        useOc8Sg4K2 = false;
    }
    DecodeGemvPlan plan;
    if (useOc8Sg4K2) {
        plan.variant = DecodeGemvVariant::Oc8Sg4K2;
        plan.tgQuads = 2;
        int lanesPerBlock = gemvLanesPerBlock(icDiv4, blockCount, true);
        if (splitKFitsBudget && oc16PreferredOverSplitK(quadsPerBlock) && 4 * oc16Threadgroups >= caps.tgBudget) {
            lanesPerBlock = 4;
        } else if (oc16ExceedsLimit && caps.tier.highBandwidth) {
            lanesPerBlock = gemvSplitKFillLanes(blockCount);
        }
        plan.lanesPerBlock = gemvCacheLineLanes(icDiv4, blockCount, shape.dequantBits, lanesPerBlock);
    } else {
        const bool useOc16 = (oc % 16 == 0) && oc16EligibleVsPlain(quadsPerBlock, caps.tier.belowGen16);
        plan.variant = useOc16 ? DecodeGemvVariant::Oc16Sg4K1 : DecodeGemvVariant::Plain;
        plan.tgQuads = useOc16 ? 4 : 2;
        plan.lanesPerBlock = gemvPlainLanesPerBlock(icDiv4, blockCount, shape.dequantBits);
    }
    return plan;
}

} // namespace MNN

#endif /* MetalConv1x1Decode_hpp */
