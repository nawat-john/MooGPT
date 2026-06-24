cmake_minimum_required(VERSION 3.16)
project(MooGPT LANGUAGES CXX)

# --- Standards & defaults ----------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Export compile_commands.json for editors/clangd.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warnings: this is a learning project, keep them loud.
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Phase 6: OpenMP parallelizes the hot loops (matmul / linear / attention). Optional —
# without it the #pragma omp lines are ignored and the build is the serial CPU path.
find_package(OpenMP)

# --- Core library (everything except main) -----------------------------------
# As phases land, add their .cpp files here (attention, mlp, model, ...).
add_library(moocore
  src/tensor.cpp
  src/ops.cpp
  src/tokenizer.cpp
  src/bpe.cpp
  src/dataloader.cpp
  src/attention.cpp
  src/mlp.cpp
  src/block.cpp
  src/model.cpp
  src/loss.cpp
  src/optimizer.cpp
)
target_include_directories(moocore PUBLIC src)
if(OpenMP_CXX_FOUND)
  target_link_libraries(moocore PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- Executable --------------------------------------------------------------
add_executable(moogpt src/main.cpp)
target_link_libraries(moogpt PRIVATE moocore)

# --- Phase 6b: optional CUDA forward path ------------------------------------
# Standalone single-TU GPU forward (src/forward_cuda.cu), not linked to moocore. Off by
# default and only built if a CUDA toolkit is present, so the CPU build never needs nvcc.
# Verify with reference/check.py exactly like `moogpt check` (see CLAUDE.md Phase 6b).
option(MOOGPT_CUDA "Build the CUDA forward path (moogpt_cuda)" OFF)
if(MOOGPT_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_executable(moogpt_cuda src/forward_cuda.cu)
    # RTX 3050 is Ampere (sm_86); 14.51 MSVC crashes nvcc, so configure with a 14.43 host
    # compiler (-DCMAKE_CUDA_HOST_COMPILER=...cl.exe) — see CLAUDE.md Phase 6b.
    set_target_properties(moogpt_cuda PROPERTIES CUDA_ARCHITECTURES 86)
    target_compile_options(moogpt_cuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler>)
  else()
    message(WARNING "MOOGPT_CUDA=ON but no CUDA compiler found; skipping moogpt_cuda.")
  endif()
endif()

# --- Tests -------------------------------------------------------------------
enable_testing()

add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE moocore)
add_test(NAME test_ops COMMAND test_ops)

add_executable(test_data tests/test_data.cpp)
target_link_libraries(test_data PRIVATE moocore)
add_test(NAME test_data COMMAND test_data)

add_executable(test_grad tests/test_grad.cpp)
target_link_libraries(test_grad PRIVATE moocore)
add_test(NAME test_grad COMMAND test_grad)

add_executable(test_optim tests/test_optim.cpp)
target_link_libraries(test_optim PRIVATE moocore)
add_test(NAME test_optim COMMAND test_optim)

add_executable(test_bpe tests/test_bpe.cpp)
target_link_libraries(test_bpe PRIVATE moocore)
add_test(NAME test_bpe COMMAND test_bpe)
