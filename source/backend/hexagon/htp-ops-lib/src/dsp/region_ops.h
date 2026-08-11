#ifndef REGION_OPS_H
#define REGION_OPS_H

#include <stdint.h>
#include <stddef.h>

// Raster region layout compatible with host side HexagonRaster::RasterRegion
typedef struct {
  int32_t srcIndex;
  int32_t srcOffset;
  int32_t dstOffset;
  int32_t size[3];
  int32_t srcStride[3];
  int32_t dstStride[3];
} __attribute__((packed)) HtpOpsRasterRegion;

typedef struct {
  int32_t src0Offset;
  int32_t src1Offset;
  int32_t dstOffset;
  int32_t size[3];
  int32_t src0Stride[3];
  int32_t src1Stride[3];
  int32_t dstStride[3];
} __attribute__((packed)) HtpOpsBinaryRegion;

typedef struct {
  int32_t loopNumber;
  int32_t sizeXYZ[3];
  int32_t dstStrideXYZ[3];
  int32_t src0StrideXYZ[3];
  int32_t src1StrideXYZ[3];
  int32_t cmdSteps[3];
  int32_t cmdViewOffset[3];
  int64_t outputElementSize;
  int64_t input0Size;
  int64_t input1Size;
  // followed by iter arrays:
  // int32_t iter0[loopNumber];
  // int32_t iter1[loopNumber];
  // int32_t iter2[loopNumber];
} __attribute__((packed)) HtpOpsLoopParam;

static inline size_t raster_region_span_bytes(const int32_t size[3], const int32_t stride[3], int bytes) {
  if (size[0] <= 0 || size[1] <= 0 || size[2] <= 0) {
    return 0;
  }
  int32_t s0 = size[0] - 1;
  int32_t s1 = size[1] - 1;
  int32_t s2 = size[2] - 1;
  int32_t maxOff = s0 * stride[0] + s1 * stride[1] + s2 * stride[2] + bytes;
  if (maxOff < 0) {
    return 0;
  }
  return (size_t) maxOff;
}

#endif
