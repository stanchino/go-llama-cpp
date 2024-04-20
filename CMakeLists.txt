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
target_sources(common PUBLIC ${_common_files})

get_target_property(_llama_transient_defines llama INTERFACE_COMPILE_DEFINITIONS)
target_compile_definitions(common PRIVATE "${_llama_transient_defines}")

add_library(${PROJECT_NAME}
        go_llama.cpp go_llama.h
        tokenizer/tokenizer.cpp tokenizer/tokenizer.h
        predictor predictor/predictor.cpp predictor/predictor.h
)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer ${CMAKE_CURRENT_SOURCE_DIR}/predictor ${llama_SOURCE_DIR} ${_common_path})

install(TARGETS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE
        common
        llama
        Threads
        ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)