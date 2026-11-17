#include "dsp/worker_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <HAP_farf.h>
#include <hexagon_protos.h>
#include <qurt.h>

// 64 KB: the grouped-causal softmax stages two score rows in L1 (2 * ATTN_FIXED_WORKSPACE_KV fp16 =
// 16 KB at KV 4096), and that frame sits inside the attention call chain.
#define MNN_WORKER_DEFAULT_STACK_SIZE (4 * 16384)
#define MNN_WORKER_MAX_THREADS        30
#define MNN_WORKER_MIN_PRIO 1
#define MNN_WORKER_MAX_PRIO 254

typedef struct MNNWorkerPool MNNWorkerPool;

typedef struct {
  MNNWorkerPool *pool;
  unsigned int index;
} MNNWorkerInfo;

// Jobs are a FIFO ring guarded by a mutex, with counting semaphores for queued jobs and free entries.
// The previous design kept workers + 1 slots in a bitmask and handed out the lowest set bit, which had two
// costs for pipelined kernels: a submitter blocked as soon as more than workers + 1 jobs were outstanding
// (a prefill GEMM queues ~20 per call), and refilled low slots were picked before older high ones, so
// jobs did not run in submission order and a kernel could not rely on that order for its dependencies.
#define MNN_WORKER_QUEUE_CAP 64

struct MNNWorkerPool {
  qurt_mutex_t      ring_mutex;
  qurt_sem_t        items;
  qurt_sem_t        spaces;
  unsigned int      head, tail;
  volatile int      stop;
  unsigned int worker_count;
  worker_pool_job_t jobs[MNN_WORKER_QUEUE_CAP];
  qurt_thread_t *threads;
  MNNWorkerInfo *infos;
  void *stack_block;
  int stack_size;
};

typedef union {
  worker_synctoken_t raw;
  struct {
    unsigned int remaining;
    unsigned int reserved;
    qurt_sem_t done;
  } state;
} MNNWorkerSyncToken;

unsigned int g_max_num_workers = 1;

static worker_pool_context_t g_default_pool = NULL;

static unsigned int mnn_worker_clamp_count(unsigned int count) {
  if (count == 0) {
    count = 1;
  }
  if (count > MNN_WORKER_MAX_THREADS) {
    count = MNN_WORKER_MAX_THREADS;
  }
  return count;
}

static int mnn_worker_clamp_priority(int priority) {
  if (priority < MNN_WORKER_MIN_PRIO) {
    return MNN_WORKER_MIN_PRIO;
  }
  if (priority > MNN_WORKER_MAX_PRIO) {
    return MNN_WORKER_MAX_PRIO;
  }
  return priority;
}

static int mnn_worker_is_pool_thread(const MNNWorkerPool *pool, unsigned int *worker_index) {
  qurt_thread_t self = qurt_thread_get_id();
  for (unsigned int i = 0; i < pool->worker_count; ++i) {
    if (pool->threads[i] == self) {
      if (worker_index != NULL) {
        *worker_index = i;
      }
      return 1;
    }
  }
  return 0;
}

static void mnn_worker_loop(void *opaque) {
  MNNWorkerInfo *info = (MNNWorkerInfo *)opaque;
  MNNWorkerPool *pool = info->pool;

  while (1) {
    (void) qurt_sem_down(&pool->items);
    if (pool->stop) {
      break;
    }
    qurt_mutex_lock(&pool->ring_mutex);
    worker_pool_job_t job = pool->jobs[pool->head % MNN_WORKER_QUEUE_CAP];
    ++pool->head;
    qurt_mutex_unlock(&pool->ring_mutex);
    (void) qurt_sem_up(&pool->spaces);

    if (job.fptr != NULL) {
      job.fptr(job.dptr, (int)info->index);
    }
  }

  qurt_thread_exit(0);
}

void worker_pool_global_init(void) {
  if (g_default_pool != NULL) {
    return;
  }

  unsigned int count = qurt_hvx_get_units() >> 8;
  if (count == 0) {
    qurt_sysenv_max_hthreads_t threads;
    if (qurt_sysenv_get_max_hw_threads(&threads) == QURT_EOK) {
      count = threads.max_hthreads;
    }
  }
  g_max_num_workers = mnn_worker_clamp_count(count);

  if (worker_pool_init(&g_default_pool) != AEE_SUCCESS) {
    // The stacks are one malloc for all workers. Retry at half size before degrading to one worker:
    // a tight DSP heap should cost a little attention stack headroom, not every worker.
    FARF(ERROR, "MNN worker pool init failed with %d KB stacks, retrying at half",
         MNN_WORKER_DEFAULT_STACK_SIZE / 1024);
    if (worker_pool_init_with_stack_size(&g_default_pool, MNN_WORKER_DEFAULT_STACK_SIZE / 2) != AEE_SUCCESS) {
      FARF(ERROR, "MNN worker pool init failed");
      g_default_pool    = NULL;
      g_max_num_workers = 1;
    }
  }
}

void worker_pool_global_deinit(void) {
  worker_pool_deinit(&g_default_pool);
}

AEEResult worker_pool_init_ex(worker_pool_context_t *context, int stack_size, int n_workers) {
  if (context == NULL || stack_size <= 0 || n_workers <= 0) {
    return AEE_EBADPARM;
  }

  unsigned int worker_count = mnn_worker_clamp_count((unsigned int)n_workers);
  MNNWorkerPool *pool = (MNNWorkerPool *)calloc(1, sizeof(MNNWorkerPool));
  if (pool == NULL) {
    return AEE_ENOMEMORY;
  }

  pool->worker_count = worker_count;
  pool->stack_size   = stack_size;
  pool->threads = (qurt_thread_t *)calloc(worker_count, sizeof(qurt_thread_t));
  pool->infos = (MNNWorkerInfo *)calloc(worker_count, sizeof(MNNWorkerInfo));
  pool->stack_block = malloc((size_t)stack_size * worker_count);

  if (pool->threads == NULL || pool->infos == NULL || pool->stack_block == NULL) {
    free(pool->threads);
    free(pool->infos);
    free(pool->stack_block);
    free(pool);
    return AEE_ENOMEMORY;
  }

  qurt_mutex_init(&pool->ring_mutex);
  qurt_sem_init_val(&pool->items, 0);
  qurt_sem_init_val(&pool->spaces, MNN_WORKER_QUEUE_CAP);

  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  int priority = mnn_worker_clamp_priority(qurt_thread_get_priority(qurt_thread_get_id()));
  char name[16];

  for (unsigned int i = 0; i < worker_count; ++i) {
    snprintf(name, sizeof(name), "mnn_w%02u", i);
    qurt_thread_attr_set_name(&attr, name);
    qurt_thread_attr_set_stack_addr(&attr, (uint8_t *)pool->stack_block + (size_t)i * stack_size);
    qurt_thread_attr_set_stack_size(&attr, stack_size);
    qurt_thread_attr_set_priority(&attr, priority);

    pool->infos[i].pool = pool;
    pool->infos[i].index = i;
    int err = qurt_thread_create(&pool->threads[i], &attr, mnn_worker_loop, &pool->infos[i]);
    if (err != QURT_EOK) {
      FARF(ERROR, "MNN worker thread create failed: %d", err);
      worker_pool_context_t cleanup = pool;
      worker_pool_deinit(&cleanup);
      return AEE_EQURTTHREADCREATE;
    }
  }

  *context = pool;
  return AEE_SUCCESS;
}

AEEResult worker_pool_init_with_stack_size(worker_pool_context_t *context, int stack_size) {
  return worker_pool_init_ex(context, stack_size, (int)g_max_num_workers);
}

AEEResult worker_pool_init(worker_pool_context_t *context) {
  return worker_pool_init_with_stack_size(context, MNN_WORKER_DEFAULT_STACK_SIZE);
}

void worker_pool_deinit(worker_pool_context_t *context) {
  if (context == NULL || *context == NULL) {
    return;
  }

  MNNWorkerPool *pool = (MNNWorkerPool *)*context;
  pool->stop          = 1;
  for (unsigned int i = 0; i < pool->worker_count; ++i) {
    (void) qurt_sem_up(&pool->items);
  }

  for (unsigned int i = 0; i < pool->worker_count; ++i) {
    if (pool->threads != NULL && pool->threads[i] != 0) {
      int status = 0;
      (void)qurt_thread_join(pool->threads[i], &status);
    }
  }

  qurt_mutex_destroy(&pool->ring_mutex);
  qurt_sem_destroy(&pool->items);
  qurt_sem_destroy(&pool->spaces);
  free(pool->threads);
  free(pool->infos);
  free(pool->stack_block);
  free(pool);
  *context = NULL;
}

AEEResult worker_pool_available(worker_pool_context_t context) {
  if (context != NULL) {
    return AEE_SUCCESS;
  }
  return g_default_pool != NULL ? AEE_SUCCESS : AEE_ERESOURCENOTFOUND;
}

AEEResult worker_pool_submit(worker_pool_context_t context, worker_pool_job_t job) {
  MNNWorkerPool *pool = (MNNWorkerPool *)(context != NULL ? context : g_default_pool);
  if (pool == NULL || job.fptr == NULL) {
    return AEE_EBADPARM;
  }

  unsigned int worker_index = 0;
  if (mnn_worker_is_pool_thread(pool, &worker_index)) {
    job.fptr(job.dptr, (int)worker_index);
    return AEE_SUCCESS;
  }

  if (pool->stop) {
    return AEE_ENOMORE;
  }
  (void) qurt_sem_down(&pool->spaces);
  qurt_mutex_lock(&pool->ring_mutex);
  pool->jobs[pool->tail % MNN_WORKER_QUEUE_CAP] = job;
  ++pool->tail;
  qurt_mutex_unlock(&pool->ring_mutex);
  (void) qurt_sem_up(&pool->items);
  return AEE_SUCCESS;
}

AEEResult worker_pool_enqueue_batch(worker_pool_context_t context, const worker_pool_job_t *jobs, unsigned int n) {
  MNNWorkerPool *pool = (MNNWorkerPool *) (context != NULL ? context : g_default_pool);
  if (pool == NULL || jobs == NULL || n > MNN_WORKER_QUEUE_CAP) {
    return AEE_EBADPARM;
  }
  if (pool->stop) {
    return AEE_ENOMORE;
  }
  for (unsigned int i = 0; i < n; ++i) {
    (void) qurt_sem_down(&pool->spaces);
  }
  qurt_mutex_lock(&pool->ring_mutex);
  for (unsigned int i = 0; i < n; ++i) {
    pool->jobs[pool->tail % MNN_WORKER_QUEUE_CAP] = jobs[i];
    ++pool->tail;
  }
  qurt_mutex_unlock(&pool->ring_mutex);
  (void) qurt_sem_add(&pool->items, n);
  return AEE_SUCCESS;
}

void worker_pool_synctoken_init(worker_synctoken_t *token, unsigned int njobs) {
  MNNWorkerSyncToken *sync = (MNNWorkerSyncToken *)token;
  sync->state.remaining = njobs;
  qurt_sem_init_val(&sync->state.done, 0);
  if (njobs == 0) {
    (void)qurt_sem_up(&sync->state.done);
  }
}

void worker_pool_synctoken_jobdone(worker_synctoken_t *token) {
  MNNWorkerSyncToken *sync = (MNNWorkerSyncToken *)token;
  if (worker_pool_atomic_dec_return(&sync->state.remaining) == 0) {
    (void)qurt_sem_up(&sync->state.done);
  }
}

void worker_pool_synctoken_wait(worker_synctoken_t *token) {
  MNNWorkerSyncToken *sync = (MNNWorkerSyncToken *)token;
  (void)qurt_sem_down(&sync->state.done);
  (void)qurt_sem_destroy(&sync->state.done);
}

AEEResult worker_pool_set_thread_priority(worker_pool_context_t context, unsigned int prio) {
  MNNWorkerPool *pool = (MNNWorkerPool *)(context != NULL ? context : g_default_pool);
  if (pool == NULL) {
    return AEE_ERESOURCENOTFOUND;
  }

  int priority = mnn_worker_clamp_priority((int)prio);
  for (unsigned int i = 0; i < pool->worker_count; ++i) {
    int err = qurt_thread_set_priority(pool->threads[i], (unsigned short)priority);
    if (err != QURT_EOK) {
      return AEE_EBADPARM;
    }
  }
  return AEE_SUCCESS;
}

AEEResult worker_pool_get_thread_priority(worker_pool_context_t context, unsigned int *prio) {
  MNNWorkerPool *pool = (MNNWorkerPool *)(context != NULL ? context : g_default_pool);
  if (pool == NULL || prio == NULL || pool->worker_count == 0) {
    return AEE_EBADPARM;
  }

  int priority = qurt_thread_get_priority(pool->threads[0]);
  if (priority <= 0) {
    *prio = 0;
    return AEE_EBADSTATE;
  }

  *prio = (unsigned int)priority;
  return AEE_SUCCESS;
}
