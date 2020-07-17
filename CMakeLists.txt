# https://devblogs.nvidia.com/building-cuda-applications-cmake/

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(AtA-gpu LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(ata-sp ${CMAKE_SOURCE_DIR}/src/ata.cu)
target_link_libraries(ata-sp ${CUDA_CUBLAS_LIBRARIES})

add_executable(ata-dp ${CMAKE_SOURCE_DIR}/src/ata.cu)
target_link_libraries(ata-dp ${CUDA_CUBLAS_LIBRARIES})
target_compile_definitions(ata-dp PUBLIC -DFLOAT_AS_DOUBLE)

set_target_properties(ata-sp ata-dp PROPERTIES CXX_STANDARD 11)

if(APPLE)
  set_property(TARGET ata-sp ata-dp PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif(APPLE)
