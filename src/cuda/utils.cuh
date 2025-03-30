__device__ int dims;
__device__ size_t size_data_per_element;
__device__ size_t offsetData;
__device__ int ef_search;
__device__ int num_data;
__device__ size_t data_size;
__device__ int max_m;

struct Node {

    __host__ __device__ Node() {
        distance = 0;
        nodeid = 0;
        checked = false;
    }

    __host__ __device__ Node(float distance, int nodeid) {
        this->distance = distance;
        this->nodeid = nodeid;
        checked = false;
    }

    __host__ __device__ Node(float distance, int nodeid, bool checked) {
        this->distance = distance;
        this->nodeid = nodeid;
        this->checked = checked;
    }

    __host__ __device__ Node(Node& other) {
        distance = other.distance;
        nodeid = other.nodeid;
        checked = other.checked;
    }

    float distance;
    int nodeid;
    bool checked;

    __host__ __device__ bool operator<(const Node &other) const {
        return distance < other.distance;
    }

    __host__ __device__ bool operator==(const Node &other) const {
        return (nodeid == other.nodeid) || (distance == other.distance);
    }

    __host__ __device__ bool operator>(const Node &other) const {
        return distance > other.distance;
    }

    __host__ __device__ bool operator>=(const Node &other) const {
        return distance >= other.distance;
    }

    __host__ __device__ bool operator<=(const Node &other) const {
        return distance <= other.distance;
    }

    __host__ __device__ bool operator!=(const Node &other) const {
        return distance != other.distance;
    }
};