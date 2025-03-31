#include "config.h"
#include <map>
#include <fstream>
#include <sstream>
#include "logger.h"

std::shared_ptr<Config> config = std::make_shared<Config>();

std::map<std::string, std::string> readConfigFile(const std::string& filename) {
    std::ifstream file(filename);
    std::map<std::string, std::string> config;
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string key, value;
            std::getline(ss, key, '=');
            std::getline(ss, value);
            config[key] = value;
        }
        file.close();
    } else {
        GlobalLogger->error("Failed to open config file: {}", filename);
        throw std::runtime_error("Failed to open config file: " + filename);
    }
    return config;
}

int Config::init(const std::string &config_file) {
    std::map<std::string, std::string> config = readConfigFile(config_file);
    dim = std::stoi(config["dim"]);
    shard_num = std::stoi(config["shard_num"]);
    memory_region_size = std::stoi(config["memory_region_size"]);
    num_data = std::stoi(config["num_data"]);
    max_m = std::stoi(config["max_m"]);
    ef_construction = std::stoi(config["ef_construction"]);
    calculate();
    return 0;
}

void Config::calculate() {
    data_size = dim * sizeof(float);
    size_data_per_element =
        4 + max_m * sizeof(unsigned int) + dim * sizeof(float) + sizeof(size_t);
    offset_data = 4 + max_m * sizeof(unsigned int);
    size_per_shard = memory_region_size / shard_num;
}