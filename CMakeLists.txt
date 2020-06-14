# https://devblogs.nvidia.com/building-cuda-applications-cmake/

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(AtA-gpu LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_library(strassen STATIC ${CMAKE_SOURCE_DIR}/src/strassen.cu)
target_compile_features(strassen PUBLIC cxx_std_11)
target_link_libraries(strassen PRIVATE ${CUDA_CUBLAS_LIBRARIES})

add_executable(ata ${CMAKE_SOURCE_DIR}/src/ata.cu)
target_link_libraries(ata PRIVATE strassen)

if(APPLE)
  set_property(TARGET ata PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif(APPLE)
