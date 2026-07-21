#ifndef HexagonFunctions_hpp
#define HexagonFunctions_hpp

struct HexagonFunctions {
    int (*execute_command_group)(int groupFd, int groupOffset, int count, int syncGroupFd, int syncGroupOffset, int syncGroupSize);
    int (*execute_command_group_profile)(int groupFd, int groupOffset, int count, int syncGroupFd, int syncGroupOffset, int syncGroupSize, int profileFd, int profileOffset, int profileSize);
};

#endif
