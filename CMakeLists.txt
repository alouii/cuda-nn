cmake_minimum_required(VERSION 3.10)
project(cuda_nn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_nn
    main.cu
    kernels.cu
)

set_target_properties(cuda_nn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)