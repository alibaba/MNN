#include <AEEStdErr.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp_capabilities_utils.h"          // $HEXAGON_SDK_ROOT/utils/examples
#include "htp_command.h"
#include "htp_ops.h"                         // QAIC auto-generated header for FastRPC

static const remote_handle64 INVALID_SESSION_HANDLE = (remote_handle64)-1;
static remote_handle64 session_handle = (remote_handle64)-1;  // global session handle
static int session_domain_id = CDSP_DOMAIN_ID;
static const char *MNN_HTP_OPS_SKEL_FALLBACK = "libMNN_htpops_skel.so";
static const char *MNN_HTP_OPS_URI_PREFIX = "file:///";
static const char *MNN_HTP_OPS_URI_SUFFIX = "?htp_ops_skel_handle_invoke&_modver=1.0";

static const char *select_htp_ops_skel(int domain_id, char *buffer, size_t buffer_size, uint32_t *expected_arch) {
  uint32_t capability = 0;
  int err = get_hex_arch_ver(domain_id, &capability);
  unsigned int arch = capability & 0xff;
  *expected_arch = 0;
  if (err != AEE_SUCCESS || arch == 0) {
    fprintf(stderr, "[MNN::Hexagon] get_hex_arch_ver failed, fallback to %s, err=0x%x capability=0x%x\n",
            MNN_HTP_OPS_SKEL_FALLBACK, err, capability);
    return MNN_HTP_OPS_SKEL_FALLBACK;
  }

  *expected_arch = arch;
  snprintf(buffer, buffer_size, "libMNN_htpops_skelV%02X.so", arch);
  fprintf(stderr, "[MNN::Hexagon] DSP arch capability=0x%x, use skel: %s\n", capability, buffer);
  return buffer;
}

static int verify_htp_ops_skel_arch(uint32_t expected_arch) {
  if (expected_arch == 0) {
    return 0;
  }

  uint32_t skel_arch = 0;
  int err = htp_ops_get_skel_arch(session_handle, &skel_arch);
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "[MNN::Hexagon] get_skel_arch failed: 0x%x\n", err);
    return -1;
  }
  if (skel_arch != expected_arch) {
    fprintf(stderr, "[MNN::Hexagon] skel arch mismatch, expected V%02X, got V%02X\n",
            expected_arch, skel_arch);
    return -1;
  }
  fprintf(stderr, "[MNN::Hexagon] skel arch verified: V%02X\n", skel_arch);
  return 0;
}

remote_handle64 get_global_handle() {
  return session_handle;
}

int open_dsp_session(int domain_id, int unsigned_pd_enabled) {
  int   err        = AEE_SUCCESS;
  char *uri_domain = NULL;

  if (session_handle != INVALID_SESSION_HANDLE) {
    return 0;
  }
  session_domain_id = domain_id;

  domain *my_domain = get_domain(domain_id);
  if (!my_domain) {
    err = AEE_EBADPARM;
    fprintf(stderr, "ERROR 0x%x: unable to get domain struct %d\n", err, domain_id);
    goto bail;
  }

  if (unsigned_pd_enabled) {
    if (&remote_session_control) {
      struct remote_rpc_control_unsigned_module ctrl;
      ctrl.domain = domain_id;
      ctrl.enable = 1;

      err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &ctrl, sizeof(ctrl));
      if (err != AEE_SUCCESS) {
        fprintf(stderr, "ERROR 0x%x: remote_session_control failed\n", err);
        goto bail;
      }
    } else {
      err = AEE_EUNSUPPORTED;
      fprintf(stderr,
              "ERROR 0x%x: remote_session_control interface is not supported on "
              "this device\n",
              err);
      goto bail;
    }
  }

  char skel_name_buffer[64];
  uint32_t expected_skel_arch = 0;
  const char *skel_name = select_htp_ops_skel(domain_id, skel_name_buffer, sizeof(skel_name_buffer),
                                             &expected_skel_arch);

  int uri_domain_len = strlen(MNN_HTP_OPS_URI_PREFIX) + strlen(skel_name) +
                       strlen(MNN_HTP_OPS_URI_SUFFIX) + MAX_DOMAIN_URI_SIZE + 1;
  uri_domain         = (char *) malloc(uri_domain_len);
  if (!uri_domain) {
    err = AEE_ENOMEMORY;
    fprintf(stderr, "unable to allocated memory for uri_domain of size: %d", uri_domain_len);
    goto bail;
  }

  err = snprintf(uri_domain, uri_domain_len, "%s%s%s%s", MNN_HTP_OPS_URI_PREFIX, skel_name,
                 MNN_HTP_OPS_URI_SUFFIX, my_domain->uri);
  if (err < 0 || err >= uri_domain_len) {
    fprintf(stderr, "ERROR 0x%x returned from snprintf\n", err);
    err = AEE_EFAILED;
    goto bail;
  }

  err = htp_ops_open(uri_domain, &session_handle);
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "DSP session open failed: 0x%08x\n", (unsigned) err);
    session_handle = INVALID_SESSION_HANDLE;
    goto bail;
  }
  if (verify_htp_ops_skel_arch(expected_skel_arch) != 0) {
    htp_ops_close(session_handle);
    session_handle = INVALID_SESSION_HANDLE;
    err = AEE_EFAILED;
    goto bail;
  }

  // enable FastRPC QoS mode
  struct remote_rpc_control_latency lat_ctrl;
  lat_ctrl.enable = RPC_PM_QOS;
  lat_ctrl.latency = 50; // target latency: 50 us (not guaranteed)

  err = remote_handle64_control(session_handle, DSPRPC_CONTROL_LATENCY, &lat_ctrl, sizeof(lat_ctrl));
  if (err) {
    fprintf(stderr, "Enabling FastRPC QoS mode failed: 0x%08x\n", (unsigned) err);
    htp_ops_close(session_handle);
    session_handle = INVALID_SESSION_HANDLE;
    goto bail;
  }

bail:
  if (uri_domain) {
    free(uri_domain);
  }
  // return err;
  return err == AEE_SUCCESS ? 0 : -1;
}

void close_dsp_session() {
  if (session_handle == INVALID_SESSION_HANDLE) {
    return;
  }
  htp_ops_close(session_handle);
  session_handle = INVALID_SESSION_HANDLE;
}

int init_htp_backend() {
  if (session_handle == INVALID_SESSION_HANDLE) {
    return -1;
  }
  int err = htp_ops_init_backend(session_handle);
  if (err != AEE_SUCCESS) {
    return -1;
  }
  return 0;
}



#include "host/op_export.h"
#include "htp_command.h"
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

int getHtpInfo(int fd, int offset) {
  return 0; // Deprecated, will be removed
}
