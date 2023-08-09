cmake_minimum_required(VERSION 3.19)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR D:/ml/opencv/build)
set(TensorRT_DIR D:/ml/TensorRT)

# rapidjson 为相对目录,可以更改为绝对目录
include_directories(include)

# opencv
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
    message("OpenCV_LIBS: " ${OpenCV_LIBS})
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# cuda https://blog.csdn.net/qq_52859223/article/details/124182319
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("CUDA include path: ${CUDA_INCLUDE_DIRS}")
    message("CUDA libraries: ${CUDA_LIBRARIES}")
endif()
include_directories(${CUDA_INCLUDE_DIRS})

# enable_language(CUDA)  # add this line, then no need to setup cuda path in vs

# TensorRT
include_directories(${TensorRT_DIR}/include)
link_directories(${TensorRT_DIR}/lib)
# link_directories($ENV{CUDA_PATH}/lib/x64)

# 链接所有库，不指定cpp文件
link_libraries(${OpenCV_LIBS}
               ${CUDA_LIBRARIES}
               # cuda.lib cublas.lib cudart.lib cudnn.lib cudnn64_8.lib
                nvinfer.lib nvinfer_plugin.lib nvonnxparser.lib nvparsers.lib)

add_executable(main main.cpp utils.cpp opencv_utils.cpp inference.hpp logger.cpp)
