#include "gpu_memory_manager.h"

std::shared_ptr<GPUMemoryManager> gpu_memory_manager;

char * GPUMemoryManager::get_shard_buffer(int shard) {
    return ((char *)buffer + size_per_shard * shard);
}