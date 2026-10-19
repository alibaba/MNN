#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*hmx_queue_callback_t)(void*);

typedef struct {
  int                  type;
  hmx_queue_callback_t callback;
  void                *data;
  unsigned int         done;
} hmx_queue_job_t;

// Caller retains the job and callback data until wait completes.
void hmx_queue_execute_async(hmx_queue_job_t *job, hmx_queue_callback_t callback, void *data);
void hmx_queue_wait(hmx_queue_job_t *job);

void hmx_queue_setup();
void hmx_queue_reset();
void hmx_queue_begin();
void hmx_queue_end();
void hmx_queue_execute(hmx_queue_callback_t callback, void* data);
void hmx_queue_execute_with_spin(hmx_queue_callback_t callback, void* data, unsigned int spin_count);

#ifdef __cplusplus
}
#endif
