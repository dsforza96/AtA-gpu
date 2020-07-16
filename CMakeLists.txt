# https://devblogs.nvidia.com/building-cuda-applications-cmake/

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(AtA-gpu LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(ata ${CMAKE_SOURCE_DIR}/src/ata.cu)
set_target_properties(ata PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_link_libraries(ata ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})
target_compile_definitions(ata PUBLIC -DFLOAT_AS_DOUBLE)

if(APPLE)
  set_property(TARGET ata PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif(APPLE)
