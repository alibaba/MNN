#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void vtcm_manager_setup();
void vtcm_manager_reset();
int vtcm_manager_acquire();
void vtcm_manager_release();
int vtcm_manager_is_acquired();
int vtcm_manager_needs_release();

void *vtcm_manager_get_vtcm_base();
int vtcm_manager_get_ctx_id();

void *vtcm_manager_reserve_area(const char *name, size_t size, size_t alignment);
void *vtcm_manager_query_area(const char *name);

static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
  // align up to 128 bytes for DMA and HVX requirements
  size_t aligned_size = (size + 127) & ~127;
  uint8_t *p = *vtcm_ptr;
  *vtcm_ptr += aligned_size;
  return p;
}

#ifdef __cplusplus
}
#endif
