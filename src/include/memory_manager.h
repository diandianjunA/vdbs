#pragma once
#include <cstddef>
#include <cstdlib>
#include <memory>

class MemoryManager {
  public:
    MemoryManager() = default;
    virtual ~MemoryManager() { free(buffer); };
    virtual void *get_buffer() { return buffer; };
    virtual void set_buffer(void *buf) { buffer = buf; };
    virtual size_t get_size() { return size; };
    virtual int allocate_memory(size_t size) {
        buffer = malloc(size);
        this->size = size;
        return 0;
    };

  private:
    void *buffer;
    size_t size;
};
