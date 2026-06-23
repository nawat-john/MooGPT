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

# --- Core library (everything except main) -----------------------------------
# As phases land, add their .cpp files here (attention, mlp, model, ...).
add_library(moocore
  src/tensor.cpp
  src/ops.cpp
  src/tokenizer.cpp
  src/dataloader.cpp
  src/attention.cpp
  src/mlp.cpp
  src/block.cpp
  src/model.cpp
  src/loss.cpp
  src/optimizer.cpp
)
target_include_directories(moocore PUBLIC src)

# --- Executable --------------------------------------------------------------
add_executable(moogpt src/main.cpp)
target_link_libraries(moogpt PRIVATE moocore)

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
