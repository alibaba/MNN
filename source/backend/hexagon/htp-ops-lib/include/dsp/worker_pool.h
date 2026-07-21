#ifndef MNN_DSP_WORKER_POOL_H
#define MNN_DSP_WORKER_POOL_H

#include <AEEStdDef.h>
#include <AEEStdErr.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WORKERPOOL_API __attribute__((visibility("default")))

typedef void (*worker_callback_t)(void *, int);
typedef void *worker_pool_context_t;

typedef struct {
  worker_callback_t fptr;
  void *dptr;
} worker_pool_job_t;

typedef struct {
  unsigned int storage[8];
} worker_synctoken_t __attribute__((aligned(8)));

WORKERPOOL_API extern unsigned int g_max_num_workers;

#define WORKER_POOL_STACK_ALLOC(type, count) ((type *)__builtin_alloca(sizeof(type) * (count)))

WORKERPOOL_API void worker_pool_global_init(void);
WORKERPOOL_API void worker_pool_global_deinit(void);
WORKERPOOL_API AEEResult worker_pool_init_ex(worker_pool_context_t *context, int stack_size, int n_workers);
WORKERPOOL_API AEEResult worker_pool_init(worker_pool_context_t *context);
WORKERPOOL_API AEEResult worker_pool_init_with_stack_size(worker_pool_context_t *context, int stack_size);
WORKERPOOL_API void worker_pool_deinit(worker_pool_context_t *context);
WORKERPOOL_API AEEResult worker_pool_available(worker_pool_context_t context);
WORKERPOOL_API AEEResult worker_pool_submit(worker_pool_context_t context, worker_pool_job_t job);
WORKERPOOL_API void worker_pool_synctoken_init(worker_synctoken_t *token, unsigned int njobs);
WORKERPOOL_API void worker_pool_synctoken_jobdone(worker_synctoken_t *token);
WORKERPOOL_API void worker_pool_synctoken_wait(worker_synctoken_t *token);
WORKERPOOL_API AEEResult worker_pool_set_thread_priority(worker_pool_context_t context, unsigned int prio);
WORKERPOOL_API AEEResult worker_pool_get_thread_priority(worker_pool_context_t context, unsigned int *prio);

static inline unsigned int worker_pool_atomic_inc_return(unsigned int *target) {
  return __atomic_add_fetch(target, 1u, __ATOMIC_ACQ_REL);
}

static inline unsigned int worker_pool_atomic_dec_return(unsigned int *target) {
  return __atomic_sub_fetch(target, 1u, __ATOMIC_ACQ_REL);
}

#ifdef __cplusplus
}
#endif

#define EXPAND_COMMON_TASK_STATE_MEMBERS \
  worker_synctoken_t sync_ctx;           \
  unsigned int       task_id;            \
  int                n_tasks;            \
  int                n_tot_chunks;       \
  int                n_chunks_per_task;

#define INIT_COMMON_TASK_STATE_MEMBERS(state, total_chunks, chunks_per_task)             \
  do {                                                                                   \
    (state).task_id = 0;                                                                 \
    (state).n_tasks = ((total_chunks) + (chunks_per_task)-1) / (chunks_per_task);        \
    (state).n_tot_chunks = (total_chunks);                                               \
    (state).n_chunks_per_task = (chunks_per_task);                                       \
  } while (0)

#endif
