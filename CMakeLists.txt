cmake_minimum_required(VERSION 3.22)

set(PROJECT_NAME tensortRT_demo)

project(${PROJECT_NAME})
add_executable(${PROJECT_NAME} main.cpp)


# 查找cuda环境
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")


# 从系统环境中寻找tensorRT相关的头文件和库
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_PARSER nvonnxparser
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

message(STATUS "TENSORRT_INCLUDE_DIR: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TENSORRT_LIBRARY_INFER: ${TENSORRT_LIBRARY_INFER}")
message(STATUS "TENSORRT_LIBRARY_INFER_PLUGIN: ${TENSORRT_LIBRARY_INFER_PLUGIN}")
message(STATUS "TENSORRT_LIBRARY_PARSER: ${TENSORRT_LIBRARY_PARSER}")

# 利用OpenMP并行化
find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
message(STATUS "OpenMP_CXX_LIBRARIES: ${OpenMP_CXX_LIBRARIES}")

# 查找OpenCV环境
find_package(OpenCV REQUIRED)
include_directories("${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")

target_link_libraries(${PROJECT_NAME} PRIVATE 
                        ${TENSORRT_LIBRARY_INFER}
                        ${TENSORRT_LIBRARY_INFER_PLUGIN}
                        ${TENSORRT_LIBRARY_PARSER}
                        ${CUDA_LIBRARIES}
                        ${OpenCV_LIBRARIES})

target_include_directories(${PROJECT_NAME} PRIVATE 
                            ${TENSORRT_INCLUDE_DIR}
                            ${OpenCV_INCLUDE_DIRS})

target_sources(${PROJECT_NAME} PRIVATE 
                            ${CMAKE_CURRENT_SOURCE_DIR}/TensortRTExecuteEngine.cpp)

