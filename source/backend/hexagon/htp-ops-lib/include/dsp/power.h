#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void power_setup();
void power_reset();
void power_acquire();
void power_release();
void power_release_all();

#ifdef __cplusplus
}
#endif
