#include "dsp/power.h"

#include <HAP_farf.h>
#include <HAP_power.h>
#if defined(__has_include)
#if __has_include(<HAP_dcvs.h>)
#include <HAP_dcvs.h>
#define MNN_HAVE_HAP_DCVS_H 1
#endif
#endif

#include <string.h>

static int power_ctx;

// TODO(hzx): maybe we should set params according to SoC model
void power_setup() {
  int err;

  HAP_power_request_t req;

  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_apptype;
  req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

  err = HAP_power_set(&power_ctx, &req);
  if (err != AEE_SUCCESS) {
    FARF(ALWAYS, "HAP_power_set app type failed with return code 0x%x", err);
  }

  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_DCVS_v3;

  req.dcvs_v3.set_dcvs_enable = TRUE;
  req.dcvs_v3.dcvs_enable = FALSE;

  req.dcvs_v3.set_sleep_disable = TRUE;
  req.dcvs_v3.sleep_disable = TRUE;

  req.dcvs_v3.set_core_params           = TRUE;
  req.dcvs_v3.core_params.min_corner    = HAP_DCVS_VCORNER_MAX;
  req.dcvs_v3.core_params.max_corner    = HAP_DCVS_VCORNER_MAX;
  req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;

  req.dcvs_v3.set_bus_params           = TRUE;
  req.dcvs_v3.bus_params.min_corner    = HAP_DCVS_VCORNER_MAX;
  req.dcvs_v3.bus_params.max_corner    = HAP_DCVS_VCORNER_MAX;
  req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;

#if (__HEXAGON_ARCH__ >= 79) && defined(MNN_HAVE_HAP_DCVS_H)
  HAP_set_dcvs_v3_protected_bus_corners(&req, 1);
#endif

  err = HAP_power_set(&power_ctx, &req);
  if (err != AEE_SUCCESS) {
    FARF(ALWAYS, "HAP_power_set DCVS v3 failed with return code 0x%x", err);
  }

  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_HVX;
  req.hvx.power_up = TRUE;

  err = HAP_power_set(&power_ctx, &req);
  if (err != AEE_SUCCESS) {
    FARF(ALWAYS, "HAP_power_set HVX failed with return code 0x%x", err);
  }

  memset(&req, 0, sizeof(req));
#if defined(__HVX_ARCH__) && (__HVX_ARCH__ >= 75)
  req.type = HAP_power_set_HMX_v2;
  req.hmx_v2.set_power     = TRUE;
  req.hmx_v2.power_up      = TRUE;
  req.hmx_v2.set_clock     = TRUE;
  req.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;
  req.hmx_v2.min_corner    = HAP_DCVS_EXP_VCORNER_MAX;
  req.hmx_v2.max_corner    = HAP_DCVS_EXP_VCORNER_MAX;
  req.hmx_v2.perf_mode     = HAP_CLK_PERF_HIGH;
#else
  req.type         = HAP_power_set_HMX;
  req.hmx.power_up = TRUE;
#endif

  err = HAP_power_set(&power_ctx, &req);
  if (err != AEE_SUCCESS) {
    FARF(ALWAYS, "HAP_power_set HMX failed with return code 0x%x", err);
  }
}

void power_reset() {
  HAP_power_request_t req;

  memset(&req, 0, sizeof(req));
#if defined(__HVX_ARCH__) && (__HVX_ARCH__ >= 75)
  req.type = HAP_power_set_HMX_v2;
  req.hmx_v2.set_power = TRUE;
  req.hmx_v2.power_up = FALSE;
#else
  req.type         = HAP_power_set_HMX;
  req.hmx.power_up = FALSE;
#endif
  HAP_power_set(&power_ctx, &req);

  HAP_power_set_dcvs_v3_init(&req);
  HAP_power_set(&power_ctx, &req);
}
