# https://devblogs.nvidia.com/building-cuda-applications-cmake/

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(AtA-gpu LANGUAGES CXX CUDA)

add_library(strassen STATIC src/strassen.cu)
target_compile_features(strassen PUBLIC cxx_std_11)
target_link_libraries(strassen PRIVATE -lcublas)

add_executable(ata src/ata.cu)
target_link_libraries(ata PRIVATE strassen)

if(APPLE)
  set_property(TARGET ata PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif(APPLE)
