#pragma once

#include <remote.h>

int open_dsp_session(int domain_id, int unsigned_pd_enabled);
void close_dsp_session();

remote_handle64 get_global_handle();

int init_htp_backend();
int htp_dspqueue_execute_command_group(int groupFd, int groupOffset, int count, int syncGroupFd, int syncGroupOffset, int syncGroupSize);
int htp_dspqueue_execute_command_group_profile(int groupFd, int groupOffset, int count, int syncGroupFd, int syncGroupOffset, int syncGroupSize, int profileFd, int profileOffset, int profileSize);

int getHtpInfo(int fd, int offset);
