#include "gpu_memory_manager.h"

std::shared_ptr<GPUMemoryManager> gpu_memory_manager;

std::pair<void *, std::shared_ptr<std::mutex>>
GPUMemoryManager::get_shard_buffer(int shard) {
    return std::make_pair((void *)((char *)buffer + size_per_shard * shard), locks[shard]);
}