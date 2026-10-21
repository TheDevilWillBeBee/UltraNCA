cmake_minimum_required(VERSION 3.24...3.27)

# the GPU of this machine (mma.sync TF32 needs sm_80 or newer); e.g. -DCMAKE_CUDA_ARCHITECTURES="80;86;89" for others
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(ncaops LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -lineinfo")

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(nca_cuda nca_cuda.cu)
# whole-program device compilation: no relocatable-code register limits
set_property(TARGET nca_cuda PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(nca_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
# per-GPU configuration and tuning overrides (see nca_step.cuh), e.g. cmake -DNCA_DEFS="-DNCA_SMEM_PER_SM_KB=164"
if (NCA_DEFS)
  separate_arguments(NCA_DEFS_LIST UNIX_COMMAND "${NCA_DEFS}")
  target_compile_options(nca_cuda PRIVATE ${NCA_DEFS_LIST})
endif()
target_include_directories(nca_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# LPPN decoder (lppn.py): its own module
nanobind_add_module(lppn_cuda lppn_cuda.cu)
set_property(TARGET lppn_cuda PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(lppn_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
target_include_directories(lppn_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
