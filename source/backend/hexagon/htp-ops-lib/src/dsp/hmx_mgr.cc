#include "dsp/hmx_mgr.h"

#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <qurt/qurt_mutex.h>

#include "dsp/vtcm_mgr.h"

#if defined(__HEXAGON_ARCH__) && (__HEXAGON_ARCH__ >= 81)
#define MNN_HMX_LOCK_EXCLUSIVE 1
#else
#define MNN_HMX_LOCK_EXCLUSIVE 0
#endif

static qurt_mutex_t hmx_mgr_mutex = QURT_MUTEX_INIT;

void hmx_manager_setup() {
  // NOTE(hzx): HMX should be already powered up in power_setup()
}

void hmx_manager_reset() {
}

void hmx_manager_enable_execution() {
  int hmx_mgr_ctx_id = vtcm_manager_get_ctx_id();
  if (!hmx_mgr_ctx_id) {
    return;
  }

#if MNN_HMX_LOCK_EXCLUSIVE
  HAP_compute_res_hmx_lock(hmx_mgr_ctx_id);
#else
  int err = HAP_compute_res_hmx_lock2(hmx_mgr_ctx_id, HAP_COMPUTE_RES_HMX_SHARED);
  if (err) {
    FARF(ALWAYS, "HAP_compute_res_hmx_lock2 failed with return code 0x%x", err);
  }
#endif
}

void hmx_manager_disable_execution() {
  int hmx_mgr_ctx_id = vtcm_manager_get_ctx_id();
  if (!hmx_mgr_ctx_id) {
    return;
  }

#if MNN_HMX_LOCK_EXCLUSIVE
  HAP_compute_res_hmx_unlock(hmx_mgr_ctx_id);
#else
  HAP_compute_res_hmx_unlock2(hmx_mgr_ctx_id, HAP_COMPUTE_RES_HMX_SHARED);
#endif
}

void hmx_unit_acquire() {
  qurt_mutex_lock(&hmx_mgr_mutex);
}

void hmx_unit_release() {
  qurt_mutex_unlock(&hmx_mgr_mutex);
}
