#pragma once
#include "memory_manager.h"
#include <cuda_runtime.h>
#include <mutex>
#include <tuple>
#include <vector>

class GPUMemoryManager : MemoryManager {
  public:
    GPUMemoryManager() = default;
    GPUMemoryManager(int shard_num, int size_per_element, int size_per_shard)
        : shard_num(shard_num), size_per_element(size_per_element), size_per_shard(size_per_shard) {};
    ~GPUMemoryManager() { cudaFree(buffer); };
    void *get_buffer() override { return buffer; };
    size_t get_size() override { return size; };
    int allocate_memory(size_t size) override {
        cudaMalloc(&buffer, size);
        cudaMemset(buffer, 0, size);
        this->size = size;
        size_per_shard = size / shard_num;
        return 0;
    };
    char* get_shard_buffer(int shard);
    int get_size_per_shard() { return size_per_shard; };
    int get_size_per_element() { return size_per_element; };

  private:
    void *buffer;
    size_t size;
    int shard_num;
    int size_per_shard;
    int size_per_element;
};

extern std::shared_ptr<GPUMemoryManager> gpu_memory_manager;