#include "dsp/mmap_mgr.h"

#include <HAP_mem.h>

struct MapInfo {
    int fd;
    void* ptr;
};

#define MAX_MMAP_CACHE_SIZE 512
#define MMAP_HOT_CACHE_SIZE 16

struct MmapManager {
    MapInfo entries[MAX_MMAP_CACHE_SIZE];
    MapInfo hot[MMAP_HOT_CACHE_SIZE];
    int count;
    int hot_next;
};

static inline void mmap_manager_remember_hot(MmapManager* manager, int fd, void* ptr) {
  int idx = manager->hot_next++;
  if (manager->hot_next >= MMAP_HOT_CACHE_SIZE) {
    manager->hot_next = 0;
  }
  manager->hot[idx].fd = fd;
  manager->hot[idx].ptr = ptr;
}

extern "C" {

MmapManager* mmap_manager_init_local() {
  MmapManager* manager = new MmapManager();
  if (manager) {
    manager->count = 0;
    manager->hot_next = 0;
    for (int i = 0; i < MMAP_HOT_CACHE_SIZE; ++i) {
      manager->hot[i].fd = -1;
      manager->hot[i].ptr = nullptr;
    }
  }
  return manager;
}

void mmap_manager_destroy_local(MmapManager* manager) {
  if (manager) {
    for (int i = 0; i < manager->count; ++i) {
      HAP_mmap_put(manager->entries[i].fd);
    }
    manager->count = 0;
    delete manager;
  }
}

void *mmap_manager_get_map(int fd) {
  void *p;
  int   err = HAP_mmap_get(fd, &p, nullptr);
  if (err) {
    return nullptr;
  }

  return p;
}

void *mmap_manager_get_map_local(MmapManager* manager, int fd) {
  if (manager == nullptr) {
    return mmap_manager_get_map(fd);
  }
  for (int i = 0; i < MMAP_HOT_CACHE_SIZE; ++i) {
    if (manager->hot[i].fd == fd) {
      return manager->hot[i].ptr;
    }
  }
  for (int i = 0; i < manager->count; ++i) {
    if (manager->entries[i].fd == fd) {
      mmap_manager_remember_hot(manager, fd, manager->entries[i].ptr);
      return manager->entries[i].ptr;
    }
  }

  void *p;
  int   err = HAP_mmap_get(fd, &p, nullptr);
  if (err) {
    return nullptr;
  }

  if (manager->count < MAX_MMAP_CACHE_SIZE) {
    int idx = manager->count++;
    manager->entries[idx].fd = fd;
    manager->entries[idx].ptr = p;
  }
  mmap_manager_remember_hot(manager, fd, p);

  return p;
}

void mmap_manager_release_all() {
}
}
