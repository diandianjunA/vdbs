#include <string>
#include <memory>

class Config {

public:
    Config() {};
    int init(const std::string &config_file);
    void calculate();
    ~Config() {};
    int dim;
    int shard_num;
    int memory_region_size;
    int num_data;
    int max_m;
    int ef_construction;
    int size_data_per_element;
    int offset_data;
    int data_size;
    int size_per_shard;
};

extern std::shared_ptr<Config> config;