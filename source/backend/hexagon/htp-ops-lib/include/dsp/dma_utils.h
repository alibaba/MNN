#pragma once

#include <hexagon_protos.h>
#include <stdbool.h>
#include <stdint.h>

#define DM0_STATUS_MASK  3
#define DM0_STATUS_IDLE  0
#define DM0_STATUS_RUN   1
#define DM0_STATUS_ERROR 2

#define DMA_DESC_DSTATE_PENDING 0
#define DMA_DESC_DSTATE_DONE    1

#define DMA_DESC_TYPE_1D 0
#define DMA_DESC_TYPE_2D 1

#define DMA_DESC_CACHEALLOC_MASK      3
#define DMA_DESC_CACHEALLOC_NONE      0
#define DMA_DESC_CACHEALLOC_WRITEONLY 1
#define DMA_DESC_CACHEALLOC_READONLY  2
#define DMA_DESC_CACHEALLOC_READWRITE 3

struct dma_desc_1d {
  uint32_t next;

  union {
    struct {
      unsigned length     : 24;
      unsigned type       : 2;
      unsigned dst_dlbc   : 1;
      unsigned src_dlbc   : 1;
      unsigned dst_bypass : 1;
      unsigned src_bypass : 1;
      unsigned ordered    : 1;
      unsigned dstate     : 1;
    } __attribute__((packed));

    uint32_t dstate_order_bypass_type_length;
  };

  uint32_t src;
  uint32_t dst;
} __attribute__((packed));

struct dma_desc_2d {
  uint32_t next;

  union {
    struct {
      unsigned length     : 24;
      unsigned type       : 2;
      unsigned dst_dlbc   : 1;
      unsigned src_dlbc   : 1;
      unsigned dst_bypass : 1;
      unsigned src_bypass : 1;
      unsigned ordered    : 1;
      unsigned dstate     : 1;
    } __attribute__((packed));

    uint32_t dstate_order_bypass_type_length;
  };

  uint32_t src;
  uint32_t dst;

  unsigned _pad1 : 24, cache_alloc : 2, _pad2 : 6;
  // assume little-endian
  uint16_t roi_width;
  uint16_t roi_height;
  uint16_t src_stride;
  uint16_t dst_stride;
  uint16_t src_width_offset;
  uint16_t dst_width_offset;
} __attribute__((packed));

typedef struct dma_desc_1d dma_desc_1d_t;
typedef struct dma_desc_2d dma_desc_2d_t;

static inline void dmstart(void *next) {
  asm volatile("release(%0):at" ::"r"(next));
  Q6_dmstart_A(next);
}

static inline void dmlink(void *cur, void *next) {
  asm volatile("release(%0):at" ::"r"(next));
  Q6_dmlink_AA(cur, next);
}

static inline uint32_t dmpoll() {
  return Q6_R_dmpoll();
}

static inline uint32_t dmwait() {
  return Q6_R_dmwait();
}

static inline bool dma_wait_for_idle() {
  uint32_t dm0_status = dmwait() & DM0_STATUS_MASK;

  return (dm0_status == DM0_STATUS_IDLE);
}

// NOTE(hzx): The submission will fail if there's current DMA request running
static inline int dma_submit_one(dma_desc_1d_t *desc) {
  if (!desc) {
    return -1;
  }

  uint32_t dm0_status = dmpoll() & DM0_STATUS_MASK;
  if (dm0_status != DM0_STATUS_IDLE) {
    return 1;  // TODO: define error code
  }

  dmstart(desc);
  dmpoll();
  return 0;
}
