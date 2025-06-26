cmake_minimum_required(VERSION 3.18)
project(cuEQ LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# Add executable
add_executable(cuEQ
    src/main.cpp
    src/audio_processor.cpp
    src/wav_io.cpp
    src/gpu_eq_processor.cu
    src/compressor.cpp
)

target_include_directories(cuEQ PRIVATE include)
target_link_libraries(cuEQ CUDA::cufft CUDA::cudart)

# Compiler flags
target_compile_options(cuEQ PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
