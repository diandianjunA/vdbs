cmake_minimum_required(VERSION 3.5) #设置cmake的最低版本
project(vdb LANGUAGES CXX) #设置项目名称 vdb
 
# 设置C++标准  基本不用改
set(CMAKE_CXX_STANDARD 20)  
# set(CMAKE_CXX_STANDARD_REQUIRED True)  

set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl") 
# set (LIBCUVS, "/home/xjs/anaconda3/pkgs/libcuvs-24.12.00-cuda12_241211_g0ce6a03_0/lib")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(src/cuda)
set (EXTRA_LIBS ${EXTRA_LIBS} gpu)

include_directories(src/include)
include_directories(src/cuda)
link_directories(/usr/local/lib)
link_directories(/usr/local/cuda/lib64)
link_directories(/usr/lib)
link_directories(/usr/lib/x86_64-linux-gnu)

# file自定义搜索源文件，塞给集合SOURCES 
file(GLOB INDEX_SERVER_SOURCES 
    src/index_server/*.cpp 
    src/log/*.cpp 
    src/rdma/rdma_common.cpp 
    src/rdma/rdma_pools.cpp 
    src/config/*.cpp
)
file(GLOB STORAGE_SERVER_SOURCES 
    src/storage_server/*.cpp 
    src/log/*.cpp 
    src/rdma/rdma_common.cpp 
    src/rdma/rdma_pool.cpp
    src/config/*.cpp
)
# file(GLOB MASTER_SERVER_SOURCES master_server/*.cpp log/*.cpp)
# file(GLOB PROXY_SERVER_SOURCES proxy_server/*.cpp log/*.cpp)


add_executable(index_server src/index_server.cpp ${INDEX_SERVER_SOURCES})
add_executable(storage_server src/storage_server.cpp ${STORAGE_SERVER_SOURCES})
# add_executable(master_server src/master_server.cpp ${MASTER_SERVER_SOURCES})
# add_executable(proxy_server src/proxy_server.cpp ${PROXY_SERVER_SOURCES})

target_link_libraries(index_server PRIVATE ${EXTRA_LIBS}
    ibverbs etcd-cpp-api cpprest cuda cudart spdlog
)
target_link_libraries(storage_server PRIVATE
    ibverbs 
    spdlog
)
# target_link_libraries(master_server PRIVATE
#     etcd-cpp-api cpprest
# )
# target_link_libraries(proxy_server PRIVATE
#     curl pthread
# )