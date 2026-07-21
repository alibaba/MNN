#pragma once

#include "htp_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int htp_ops_rpc_init_backend();
int htp_ops_rpc_getInfo(int fd, int offset);
int htp_ops_rpc_getInfoProfile(int fd, int offset);

int htp_rpc_execute_command_group(int groupFd, int groupOffset, int count, int syncGroupFd, int syncGroupOffset, int syncGroupSize);
int htp_rpc_execute_command_group_profile(int groupFd, int groupOffset, int count, int syncGroupFd, int syncGroupOffset, int syncGroupSize, int profileFd, int profileOffset, int profileSize);

#ifdef __cplusplus
}
#endif
