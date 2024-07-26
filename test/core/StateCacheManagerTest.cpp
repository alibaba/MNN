#include "MNNTestSuite.h"
#include "core/StateCacheManager.hpp"
#include <iostream>

using namespace PagedAttention;

#ifndef _MSC_VER
class StateCacheManagerTest : public MNNTestCase {
public:
    virtual ~StateCacheManagerTest() = default;

    static void enlargeMemCacheTest(StateCacheManager& manager, size_t size) {
        bool result = manager.enlargeMemCache(size);
        std::cout << "Enlarge Mem Cache Test (" << size << "): " << (result ? "Success" : "Failure") << std::endl;
    }

    static void enlargeFileCacheTest(StateCacheManager& manager, size_t size) {
        bool result = manager.enlargeFileCache(size);
        std::cout << "Enlarge File Cache Test (" << size << "): " << (result ? "Success" : "Failure") << std::endl;
    }

    static void releaseMemCacheTest(StateCacheManager& manager) {
        manager.releasesMemCache();
        std::cout << "Release Mem Cache Test: Success" << std::endl;
    }

    static void releaseFileCacheTest(StateCacheManager& manager) {
        manager.releaseFileCache();
        std::cout << "Release File Cache Test: Success" << std::endl;
    }

    static void evictBlockTest(StateCacheManager& manager, const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list) {
        auto evicted_block = manager.evictBlock(pin_block_list);
        if (evicted_block) {
            std::cout << "Evict Block Test: Success" << std::endl;
        } else {
            std::cout << "Evict Block Test: Failure" << std::endl;
        }
    }

    static void getFreePtrTest(StateCacheManager& manager, const std::vector<std::shared_ptr<StateCacheBlock>>& evict_pin_block_list) {
        auto free_ptr = manager.getFreePtr(evict_pin_block_list);
        if (free_ptr) {
            std::cout << "Get Free Ptr Test: Success" << std::endl;
        } else {
            std::cout << "Get Free Ptr Test: Failure" << std::endl;
        }
    }

    static void recoverBlockTest(StateCacheManager& manager, std::shared_ptr<StateCacheBlock> block_ptr, const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list) {
        manager.recoverBlock(block_ptr, pin_block_list);
        std::cout << "Recover Block Test: Success" << std::endl;
    }

    static void desertBlockTest(StateCacheManager& manager, int ref_id, std::shared_ptr<StateCacheBlock> block_ptr) {
        manager.desertBlock(ref_id, block_ptr);
        std::cout << "Desert Block Test: Success" << std::endl;
    }

    static void copyBlockTest(StateCacheManager& manager, int ref_id, std::shared_ptr<StateCacheBlock> block_ptr) {
        auto copied_block = manager.copyBlock(ref_id, block_ptr);
        if (copied_block && copied_block->ref_ids[0] == ref_id) {
            std::cout << "Copy Block Test: Success" << std::endl;
        } else {
            std::cout << "Copy Block Test: Failure" << std::endl;
        }
    }

    static void prepareAttnTest(StateCacheManager& manager, int ref_id, const std::vector<std::shared_ptr<StateCacheBlock>>& argv) {
        manager.prepareAttn(ref_id, argv);
        std::cout << "Prepare Attn Test: Success" << std::endl;
    }


    

    virtual bool run(int precision) {
        StateCacheManager manager;

        // Test case 1: Enlarge memory cache
        {
            enlargeMemCacheTest(manager, 1024 * 1024); // Enlarge by 1 MB
        }

        // Test case 2: Enlarge file cache
        {
            enlargeFileCacheTest(manager, 1024 * 1024 * 10); // Enlarge by 10 MB
        }

        // Test case 3: Release memory cache
        {
            releaseMemCacheTest(manager);
        }

        // Test case 4: Release file cache
        {
            releaseFileCacheTest(manager);
        }

        // Test case 5: Evict a block
        {
            std::vector<std::shared_ptr<StateCacheBlock>> pin_block_list;
            evictBlockTest(manager, pin_block_list);
        }

        // Test case 6: Get a free pointer
        {
            std::vector<std::shared_ptr<StateCacheBlock>> evict_pin_block_list;
            getFreePtrTest(manager, evict_pin_block_list);
        }

        // Test case 7: Recover a block
        {
            std::shared_ptr<StateCacheBlock> block_ptr = std::make_shared<StateCacheBlock>();
            std::vector<std::shared_ptr<StateCacheBlock>> pin_block_list;
            recoverBlockTest(manager, block_ptr, pin_block_list);
        }

        // Test case 8: Desert a block
        {
            int ref_id = 1;
            std::shared_ptr<StateCacheBlock> block_ptr = std::make_shared<StateCacheBlock>();
            block_ptr->ref_ids.push_back(ref_id);
            desertBlockTest(manager, ref_id, block_ptr);
        }

        // Test case 9: Copy a block
        {
            int ref_id = 2;
            std::shared_ptr<StateCacheBlock> block_ptr = std::make_shared<StateCacheBlock>();
            copyBlockTest(manager, ref_id, block_ptr);
        }

        // Test case 10: Prepare attention
        {
            int ref_id = 3;
            std::vector<std::shared_ptr<StateCacheBlock>> argv;
            prepareAttnTest(manager, ref_id, argv);
        }

        

        return true;
    }
};
MNNTestSuiteRegister(StateCacheManagerTest, "core/state_cache_manager");
#endif