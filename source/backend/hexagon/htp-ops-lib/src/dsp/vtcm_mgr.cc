#include "dsp/vtcm_mgr.h"
#include "flatbuffers/flatbuffers.h"
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <qurt_thread.h>

#include <cstring>
#include <string>
#include <unordered_map>

namespace vtcm_manager {

uint8_t *vtcm_base           = nullptr;
uint8_t *vtcm_reserved_start = nullptr;
unsigned int vtcm_total_size = 0;

int vtcm_mgr_ctx_id = 0;
bool vtcm_valid = false;
bool vtcm_needs_release = false;

std::unordered_map<std::string, uint8_t *> reserved_areas;

}  // namespace vtcm_manager

extern "C" {

static int vtcm_manager_release_callback(unsigned int rctx, void *state) {
  (void)rctx;
  (void)state;
  vtcm_manager::vtcm_needs_release = true;
  return 0;
}

void vtcm_manager_setup() {
  using namespace vtcm_manager;

  int err;

  unsigned int            avail_size, total_size;
  compute_res_vtcm_page_t avail_pages, total_pages;
  err = HAP_compute_res_query_VTCM(0, &total_size, &total_pages, &avail_size, &avail_pages);
  if (err) {
    FARF(ALWAYS, "HAP_compute_res_query_VTCM failed with return code 0x%x", err);
    return;
  }
  FARF(ALWAYS, "available VTCM size: %d KiB, total VTCM size: %d KiB", avail_size / 1024, total_size / 1024);

  vtcm_total_size = total_size;
  if (vtcm_mgr_ctx_id != 0) {
    return;
  }

  compute_res_attr_t req;
  HAP_compute_res_attr_init(&req);

  HAP_compute_res_attr_set_serialize(&req, 0);
  HAP_compute_res_attr_set_cache_mode(&req, 1);
  HAP_compute_res_attr_set_vtcm_param_v2(&req, total_size, total_size, total_size);
  HAP_compute_res_attr_set_release_callback(&req, vtcm_manager_release_callback, nullptr);
  HAP_compute_res_attr_set_hmx_param(&req, 1);

  vtcm_mgr_ctx_id = HAP_compute_res_acquire(&req, 1000000);
  if (vtcm_mgr_ctx_id == 0) {
    FARF(ALWAYS, "%s: HAP_compute_res_acquire failed", __func__);
    return;
  }

  void *vtcm_ptr = nullptr;
  if (HAP_compute_res_attr_get_vtcm_ptr_v2(&req, &vtcm_ptr, &vtcm_total_size) != 0) {
    FARF(ALWAYS, "%s: HAP_compute_res_attr_get_vtcm_ptr_v2 failed", __func__);
    HAP_compute_res_release(vtcm_mgr_ctx_id);
    vtcm_mgr_ctx_id = 0;
    vtcm_total_size = 0;
    return;
  }

  vtcm_base = (uint8_t *)vtcm_ptr;
  vtcm_reserved_start = vtcm_base + vtcm_total_size;
  vtcm_valid = false;
  reserved_areas.clear();
}

int vtcm_manager_acquire() {
  using namespace vtcm_manager;

  if (vtcm_mgr_ctx_id == 0) {
    vtcm_manager_setup();
  }
  if (vtcm_mgr_ctx_id == 0) {
    return -1;
  }

  if (!vtcm_valid) {
    int err = HAP_compute_res_acquire_cached(vtcm_mgr_ctx_id, 10000000);
    if (err != 0) {
      FARF(ALWAYS, "%s: HAP_compute_res_acquire_cached failed with return code 0x%x", __func__, err);
      return err;
    }
    vtcm_valid = true;
    vtcm_needs_release = false;

    int prio = qurt_thread_get_priority(qurt_thread_get_id());
    if (prio > 0) {
      HAP_compute_res_update_priority(vtcm_mgr_ctx_id, prio + 10);
    }
  }

  vtcm_reserved_start = vtcm_base + vtcm_total_size;
  return 0;
}

void vtcm_manager_reset() {
  using namespace vtcm_manager;

  vtcm_manager_release();
  if (vtcm_mgr_ctx_id != 0) {
    HAP_compute_res_release(vtcm_mgr_ctx_id);
    vtcm_mgr_ctx_id = 0;
  }
  vtcm_base = nullptr;
  vtcm_reserved_start = nullptr;
  vtcm_total_size = 0;
  reserved_areas.clear();
}

void vtcm_manager_release() {
  using namespace vtcm_manager;

  if (vtcm_mgr_ctx_id != 0 && vtcm_valid) {
    HAP_compute_res_release_cached(vtcm_mgr_ctx_id);
    vtcm_valid = false;
    vtcm_needs_release = false;
  }
}

int vtcm_manager_is_acquired() {
  return vtcm_manager::vtcm_valid ? 1 : 0;
}

int vtcm_manager_needs_release() {
  return vtcm_manager::vtcm_needs_release ? 1 : 0;
}

void *vtcm_manager_get_vtcm_base() {
  return vtcm_manager::vtcm_base;
}

int vtcm_manager_get_ctx_id() {
  return vtcm_manager::vtcm_mgr_ctx_id;
}

void *vtcm_manager_reserve_area(const char *name, size_t size, size_t alignment) {
  using namespace vtcm_manager;

  if (!name || (alignment & (alignment - 1)) != 0) {
    return nullptr;
  }

  std::string ident = name;
  auto        it    = reserved_areas.find(ident);
  if (it != reserved_areas.end()) {
    return it->second;
  }

  uintptr_t start_val = reinterpret_cast<uintptr_t>(vtcm_reserved_start - size) & ~(alignment - 1);
  uint8_t  *new_start = reinterpret_cast<uint8_t *>(start_val);
  if (new_start <= vtcm_base) {
    return nullptr;  // no enough space left
  }

  vtcm_reserved_start   = new_start;
  reserved_areas[ident] = new_start;
  return new_start;
}

void *vtcm_manager_query_area(const char *name) {
  using namespace vtcm_manager;

  if (!name) {
    return nullptr;
  }

  std::string ident = name;
  auto        it    = reserved_areas.find(ident);
  if (it == reserved_areas.end()) {
    return nullptr;
  }
  return it->second;
}
}
