#include "dsp/hmx_queue.h"

#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <qurt.h>
#include <qurt_futex.h>
#include <stdlib.h>

#include "dsp/hmx_mgr.h"
#include "dsp/vtcm_mgr.h"

#define MNN_HMX_QUEUE_CAPACITY 16
#define MNN_HMX_QUEUE_STACK_SIZE (2 * 16384)

#if __HVX_ARCH__ > 79
#define MNN_HMX_QUEUE_POLL_COUNT 2000
#else
#define MNN_HMX_QUEUE_POLL_COUNT 0
#endif

typedef enum {
  MNN_HMX_QUEUE_JOB_CALLBACK = 0,
  MNN_HMX_QUEUE_JOB_BEGIN = 1,
  MNN_HMX_QUEUE_JOB_END = 2,
  MNN_HMX_QUEUE_JOB_STOP = 3,
} MNNHmxQueueJobType;

typedef struct {
  MNNHmxQueueJobType type;
  hmx_queue_callback_t callback;
  void* data;
  unsigned int done;
} MNNHmxQueueJob;

typedef struct {
  qurt_thread_t thread;
  qurt_mutex_t mutex;
  qurt_sem_t free_slots;
  MNNHmxQueueJob* jobs[MNN_HMX_QUEUE_CAPACITY];
  unsigned int read_index;
  unsigned int write_index;
  unsigned int sequence;
  unsigned int active_depth;
  void* stack;
} MNNHmxQueue;

static MNNHmxQueue* g_hmx_queue = NULL;

static void hmx_queue_resource_begin(MNNHmxQueue* queue) {
  if (queue->active_depth++ != 0) {
    return;
  }
  int context_id = vtcm_manager_get_ctx_id();
  if (context_id == 0) {
    return;
  }
  HAP_compute_res_hmx_lock(context_id);
  hmx_unit_acquire();
}

static void hmx_queue_resource_end(MNNHmxQueue* queue) {
  if (queue->active_depth == 0 || --queue->active_depth != 0) {
    return;
  }
  int context_id = vtcm_manager_get_ctx_id();
  if (context_id == 0) {
    return;
  }
  hmx_unit_release();
  HAP_compute_res_hmx_unlock(context_id);
}

static void hmx_queue_thread(void* opaque) {
  MNNHmxQueue* queue = (MNNHmxQueue*)opaque;
  unsigned int observed_sequence = __atomic_load_n(&queue->sequence, __ATOMIC_ACQUIRE);
  unsigned int poll_count = MNN_HMX_QUEUE_POLL_COUNT;
  while (1) {
    while (queue->read_index == __atomic_load_n(&queue->write_index, __ATOMIC_ACQUIRE)) {
      unsigned int sequence = __atomic_load_n(&queue->sequence, __ATOMIC_ACQUIRE);
      if (sequence != observed_sequence) {
        observed_sequence = sequence;
        poll_count = MNN_HMX_QUEUE_POLL_COUNT;
        continue;
      }
      if (poll_count > 0) {
        --poll_count;
        asm volatile("pause(#8)" ::: "memory");
        continue;
      }
      (void)qurt_futex_wait(&queue->sequence, observed_sequence);
      poll_count = MNN_HMX_QUEUE_POLL_COUNT;
    }

    MNNHmxQueueJob* job = queue->jobs[queue->read_index];
    queue->read_index = (queue->read_index + 1) % MNN_HMX_QUEUE_CAPACITY;
    (void)qurt_sem_up(&queue->free_slots);

    if (job->type == MNN_HMX_QUEUE_JOB_BEGIN) {
      hmx_queue_resource_begin(queue);
    } else if (job->type == MNN_HMX_QUEUE_JOB_END) {
      hmx_queue_resource_end(queue);
    } else if (job->type == MNN_HMX_QUEUE_JOB_CALLBACK) {
      const int acquire_for_job = queue->active_depth == 0;
      if (acquire_for_job) {
        hmx_queue_resource_begin(queue);
      }
      job->callback(job->data);
      if (acquire_for_job) {
        hmx_queue_resource_end(queue);
      }
    }

    const int stop = job->type == MNN_HMX_QUEUE_JOB_STOP;
    if (stop) {
      while (queue->active_depth > 0) {
        hmx_queue_resource_end(queue);
      }
    }
    __atomic_store_n(&job->done, 1u, __ATOMIC_RELEASE);
    (void)qurt_futex_wake(&job->done, 1);
    if (stop) {
      break;
    }
  }
  qurt_thread_exit(0);
}

static void hmx_queue_submit(MNNHmxQueueJobType type, hmx_queue_callback_t callback, void* data,
                             unsigned int spin_count) {
  MNNHmxQueue* queue = g_hmx_queue;
  if (queue == NULL) {
    if (type == MNN_HMX_QUEUE_JOB_CALLBACK && callback != NULL) {
      hmx_manager_enable_execution();
      hmx_unit_acquire();
      callback(data);
      hmx_unit_release();
      hmx_manager_disable_execution();
    }
    return;
  }

  MNNHmxQueueJob job;
  job.type = type;
  job.callback = callback;
  job.data = data;
  __atomic_store_n(&job.done, 0u, __ATOMIC_RELAXED);

  (void)qurt_sem_down(&queue->free_slots);
  qurt_mutex_lock(&queue->mutex);
  unsigned int write_index = __atomic_load_n(&queue->write_index, __ATOMIC_RELAXED);
  queue->jobs[write_index] = &job;
  __atomic_store_n(&queue->write_index, (write_index + 1) % MNN_HMX_QUEUE_CAPACITY, __ATOMIC_RELEASE);
  qurt_mutex_unlock(&queue->mutex);
  (void)__atomic_add_fetch(&queue->sequence, 1u, __ATOMIC_RELEASE);
  (void)qurt_futex_wake(&queue->sequence, 1);

  for (unsigned int i = 0; i < spin_count; ++i) {
    if (__atomic_load_n(&job.done, __ATOMIC_ACQUIRE) != 0u) {
      return;
    }
    asm volatile("pause(#8)" ::: "memory");
  }
  while (__atomic_load_n(&job.done, __ATOMIC_ACQUIRE) == 0u) {
    (void)qurt_futex_wait(&job.done, 0u);
  }
}

void hmx_queue_setup() {
  if (g_hmx_queue != NULL) {
    return;
  }
  MNNHmxQueue* queue = (MNNHmxQueue*)calloc(1, sizeof(MNNHmxQueue));
  if (queue == NULL) {
    FARF(ERROR, "HMX queue allocation failed");
    return;
  }
  queue->stack = malloc(MNN_HMX_QUEUE_STACK_SIZE);
  if (queue->stack == NULL) {
    FARF(ERROR, "HMX queue stack allocation failed");
    free(queue);
    return;
  }

  qurt_mutex_init(&queue->mutex);
  qurt_sem_init_val(&queue->free_slots, MNN_HMX_QUEUE_CAPACITY);

  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_name(&attr, "mnn_hmx_queue");
  qurt_thread_attr_set_stack_addr(&attr, queue->stack);
  qurt_thread_attr_set_stack_size(&attr, MNN_HMX_QUEUE_STACK_SIZE);
  qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));

  g_hmx_queue = queue;
  int error = qurt_thread_create(&queue->thread, &attr, hmx_queue_thread, queue);
  if (error != QURT_EOK) {
    FARF(ERROR, "HMX queue thread creation failed: %d", error);
    g_hmx_queue = NULL;
    qurt_sem_destroy(&queue->free_slots);
    qurt_mutex_destroy(&queue->mutex);
    free(queue->stack);
    free(queue);
  }
}

void hmx_queue_reset() {
  MNNHmxQueue* queue = g_hmx_queue;
  if (queue == NULL) {
    return;
  }
  hmx_queue_submit(MNN_HMX_QUEUE_JOB_STOP, NULL, NULL, 0);
  int status = 0;
  (void)qurt_thread_join(queue->thread, &status);
  g_hmx_queue = NULL;
  qurt_sem_destroy(&queue->free_slots);
  qurt_mutex_destroy(&queue->mutex);
  free(queue->stack);
  free(queue);
}

void hmx_queue_begin() {
  hmx_queue_submit(MNN_HMX_QUEUE_JOB_BEGIN, NULL, NULL, 0);
}

void hmx_queue_end() {
  hmx_queue_submit(MNN_HMX_QUEUE_JOB_END, NULL, NULL, 0);
}

void hmx_queue_execute(hmx_queue_callback_t callback, void* data) {
  hmx_queue_submit(MNN_HMX_QUEUE_JOB_CALLBACK, callback, data, 0);
}

void hmx_queue_execute_with_spin(hmx_queue_callback_t callback, void* data, unsigned int spin_count) {
  hmx_queue_submit(MNN_HMX_QUEUE_JOB_CALLBACK, callback, data, spin_count);
}
