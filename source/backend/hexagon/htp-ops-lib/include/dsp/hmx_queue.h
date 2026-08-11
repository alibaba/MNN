#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*hmx_queue_callback_t)(void*);

void hmx_queue_setup();
void hmx_queue_reset();
void hmx_queue_begin();
void hmx_queue_end();
void hmx_queue_execute(hmx_queue_callback_t callback, void* data);
void hmx_queue_execute_with_spin(hmx_queue_callback_t callback, void* data, unsigned int spin_count);

#ifdef __cplusplus
}
#endif
