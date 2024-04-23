cmake_minimum_required(VERSION 3.27)
project(gollama)

if(MSVC)
    add_definitions(-D_WIN32_WINNT=0x600)
endif()

set(CMAKE_CXX_STANDARD 11)
set(ABSL_PROPAGATE_CXX_STD ON)
set(LLAMA_METAL_EMBED_LIBRARY ON)
# Threads
find_package(Threads REQUIRED)

# llama.cpp
include(FetchContent)
FetchContent_Declare(
        Llama
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp
)
FetchContent_MakeAvailable(Llama)
set(_common_path "${llama_SOURCE_DIR}/common")
file(GLOB _common_files
        "${_common_path}/*.h"
        "${_common_path}/*.cpp"
)
target_sources(common PRIVATE ${_common_files})

get_target_property(_llama_transient_defines llama INTERFACE_COMPILE_DEFINITIONS)
target_compile_definitions(common PRIVATE "${_llama_transient_defines}")

add_library(${PROJECT_NAME}
        pkg/llama/llama.cpp pkg/llama/llama.h
        pkg/options/options.cpp pkg/options/options.h
        pkg/util/util.cpp pkg/util/util.h
        pkg/tokenizer/tokenizer.cpp pkg/tokenizer/tokenizer.h
        pkg/predictor/predictor.cpp pkg/predictor/predictor.h
)
target_sources(${PROJECT_NAME} PRIVATE pkg/llama pkg/options pkg/predictor pkg/tokenizer pkg/util)
target_include_directories(${PROJECT_NAME} PRIVATE
        ${llama_SOURCE_DIR} ${_common_path})

install(TARGETS ${PROJECT_NAME} llama ggml common)

target_link_libraries(${PROJECT_NAME} PRIVATE
        common
        llama
        Threads
        ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)