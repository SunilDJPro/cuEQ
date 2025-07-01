cmake_minimum_required(VERSION 3.18)
project(cuEQProcessor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# Add executable
add_executable(cuEQProcessor
    src/main.cpp
    src/audio_processor.cpp
    src/wav_io.cpp
    src/cuEQ_processor.cu
    src/compressor.cpp
)

target_include_directories(cuEQProcessor PRIVATE include)
target_link_libraries(cuEQProcessor CUDA::cufft CUDA::cudart)

# Compiler flags
target_compile_options(cuEQProcessor PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
