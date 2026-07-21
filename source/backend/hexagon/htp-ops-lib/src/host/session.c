#include <AEEStdErr.h>
#include <dspqueue.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp_capabilities_utils.h"  // $HEXAGON_SDK_ROOT/utils/examples
#include "htp_command.h"
#include "htp_ops.h"                 // QAIC auto-generated header for FastRPC

static const remote_handle64 INVALID_SESSION_HANDLE = (remote_handle64) -1;
static remote_handle64       session_handle         = (remote_handle64) -1;  // global session handle
static dspqueue_t            session_queue          = NULL;
static int                   session_domain_id      = CDSP_DOMAIN_ID;
static unsigned int          session_queue_req_id   = 1;
static const char           *MNN_HTP_OPS_URI = "file:///libMNN_htpops_skel.so?htp_ops_skel_handle_invoke&_modver=1.0";
static const uint32_t        MNN_DSPQUEUE_TIMEOUT_US = 10000000;

remote_handle64 get_global_handle() {
  return session_handle;
}

static int htp_dspqueue_start(int domain_id) {
  if (session_queue != NULL) {
    return 0;
  }

  dspqueue_t queue = NULL;
  int        err   = dspqueue_create(domain_id, 0, sizeof(struct DSPQueueCommandGroupReq) * 8 + 1024,
                                     sizeof(struct DSPQueueCommandGroupRsp) * 8 + 1024, NULL, NULL, NULL, &queue);
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "dspqueue_create failed: 0x%08x\n", (unsigned) err);
    return -1;
  }

  uint64_t queue_id = 0;
  err               = dspqueue_export(queue, &queue_id);
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "dspqueue_export failed: 0x%08x\n", (unsigned) err);
    dspqueue_close(queue);
    return -1;
  }

  err = htp_ops_start_queue(session_handle, queue_id);
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "htp_ops_start_queue failed: 0x%08x\n", (unsigned) err);
    dspqueue_close(queue);
    return -1;
  }

  session_queue        = queue;
  session_queue_req_id = 1;
  return 0;
}

static void htp_dspqueue_stop() {
  if (session_queue == NULL) {
    return;
  }
  if (session_handle != INVALID_SESSION_HANDLE) {
    htp_ops_stop_queue(session_handle);
  }
  dspqueue_close(session_queue);
  session_queue = NULL;
}

static int htp_dspqueue_execute(const struct DSPQueueCommandGroupReq *req) {
  if (session_queue == NULL || req == NULL) {
    return -1;
  }

  int err = dspqueue_write(session_queue, 0, 0, NULL, sizeof(*req), (const uint8_t *) req, MNN_DSPQUEUE_TIMEOUT_US);
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "dspqueue_write failed: 0x%08x\n", (unsigned) err);
    return -1;
  }

  struct DSPQueueCommandGroupRsp rsp;
  memset(&rsp, 0, sizeof(rsp));
  uint32_t flags    = 0;
  uint32_t rsp_size = sizeof(rsp);
  uint32_t n_dbufs  = 0;
  err               = dspqueue_read(session_queue, &flags, 0, &n_dbufs, NULL, sizeof(rsp), &rsp_size, (uint8_t *) &rsp,
                                    MNN_DSPQUEUE_TIMEOUT_US);
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "dspqueue_read failed: 0x%08x\n", (unsigned) err);
    return -1;
  }
  if (rsp_size != sizeof(rsp) || rsp.id != req->id) {
    fprintf(stderr, "dspqueue response mismatch: size=%u rsp.id=%u req.id=%u\n", (unsigned) rsp_size, rsp.id, req->id);
    return -1;
  }
  return rsp.status;
}

int htp_dspqueue_execute_command_group(int groupFd, int groupOffset, int count, int syncGroupFd, int syncGroupOffset,
                                       int syncGroupSize) {
  if (session_queue == NULL) {
    return htp_ops_execute_command_group(session_handle, groupFd, groupOffset, count, syncGroupFd, syncGroupOffset,
                                         syncGroupSize);
  }
  struct DSPQueueCommandGroupReq req;
  memset(&req, 0, sizeof(req));
  req.id              = session_queue_req_id++;
  req.profile         = 0;
  req.groupFd         = groupFd;
  req.groupOffset     = groupOffset;
  req.count           = count;
  req.syncGroupFd     = syncGroupFd;
  req.syncGroupOffset = syncGroupOffset;
  req.syncGroupSize   = syncGroupSize;
  req.profileFd       = -1;
  return htp_dspqueue_execute(&req);
}

int htp_dspqueue_execute_command_group_profile(int groupFd, int groupOffset, int count, int syncGroupFd,
                                               int syncGroupOffset, int syncGroupSize, int profileFd, int profileOffset,
                                               int profileSize) {
  if (session_queue == NULL) {
    return htp_ops_execute_command_group_profile(session_handle, groupFd, groupOffset, count, syncGroupFd,
                                                 syncGroupOffset, syncGroupSize, profileFd, profileOffset, profileSize);
  }
  struct DSPQueueCommandGroupReq req;
  memset(&req, 0, sizeof(req));
  req.id              = session_queue_req_id++;
  req.profile         = 1;
  req.groupFd         = groupFd;
  req.groupOffset     = groupOffset;
  req.count           = count;
  req.syncGroupFd     = syncGroupFd;
  req.syncGroupOffset = syncGroupOffset;
  req.syncGroupSize   = syncGroupSize;
  req.profileFd       = profileFd;
  req.profileOffset   = profileOffset;
  req.profileSize     = profileSize;
  return htp_dspqueue_execute(&req);
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

  int uri_domain_len = strlen(MNN_HTP_OPS_URI) + MAX_DOMAIN_URI_SIZE;
  uri_domain         = (char *) malloc(uri_domain_len);
  if (!uri_domain) {
    err = AEE_ENOMEMORY;
    fprintf(stderr, "unable to allocated memory for uri_domain of size: %d", uri_domain_len);
    goto bail;
  }

  err = snprintf(uri_domain, uri_domain_len, "%s%s", MNN_HTP_OPS_URI, my_domain->uri);
  if (err < 0) {
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

  // enable FastRPC QoS mode
  struct remote_rpc_control_latency lat_ctrl;
  lat_ctrl.enable  = RPC_PM_QOS;
  lat_ctrl.latency = 50;  // target latency: 50 us (not guaranteed)

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
  htp_dspqueue_stop();
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
  if (htp_dspqueue_start(session_domain_id) != 0) {
    fprintf(stderr, "dspqueue unavailable, falling back to FastRPC execute\n");
  }
  return 0;
}

#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "host/op_export.h"
#include "htp_command.h"

int getHtpInfo(int fd, int offset) {
  return 0;  // Deprecated, will be removed
}
