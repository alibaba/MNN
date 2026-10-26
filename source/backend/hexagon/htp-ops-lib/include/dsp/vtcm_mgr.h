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
unsigned int vtcm_manager_get_vtcm_size();
int vtcm_manager_get_ctx_id();

void *vtcm_manager_reserve_area(const char *name, size_t size, size_t alignment);
void *vtcm_manager_query_area(const char *name);

// Bumped by vtcm_seq_alloc when a request would run past the end of VTCM. Not reset per command:
// htp_vtcm_overflow_count() is sampled before and after each command so the group can report which
// one overflowed. Plain 32-bit counters -- workers race on them, but every racing path only ever
// increases them and the check is "did it change", so a lost update cannot turn a failure into a pass.
unsigned int htp_vtcm_overflow_count(void);
unsigned int htp_vtcm_overflow_worst_bytes(void);
void         htp_vtcm_overflow_note(size_t overshoot);

static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
  // align up to 128 bytes for DMA and HVX requirements
  size_t aligned_size = (size + 127) & ~127;
  uint8_t *p = *vtcm_ptr;
  // This is an unchecked bump allocator and no caller looks at the result, so running past the end
  // used to corrupt whatever follows VTCM -- which reboots the whole device, not just the cDSP.
  // Clamp back into the region instead: the op then computes garbage (its buffers alias each other)
  // but stays inside VTCM, and the command group turns the recorded overflow into an error code.
  // Returning NULL is not an option precisely because callers do not check.
  uint8_t *const base         = (uint8_t *) vtcm_manager_get_vtcm_base();
  const size_t   total        = (size_t) vtcm_manager_get_vtcm_size();
  if (base != NULL && total != 0) {
    const size_t used = (size_t) (p - base);
    if (used > total || aligned_size > total - used) {
      htp_vtcm_overflow_note(used + aligned_size - total);
      return base;
    }
  }
  *vtcm_ptr += aligned_size;
  return p;
}

#ifdef __cplusplus
}
#endif
