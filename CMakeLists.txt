# =============================================================================
# Pixel Stretch TOP — TouchDesigner Custom Operator (CUDA)
#
# Produces PixelStretchTOP.dll, a CUDA-mode C++ TOP. Validated with:
#   - TouchDesigner 2025.32050 (TOP C++ API v12)
#   - CUDA Toolkit 13.3 (CUB/CCCL 3.3)
#   - Visual Studio 2022/2026 (MSVC 14.4x), CMake >= 3.24
# =============================================================================
cmake_minimum_required(VERSION 3.24)

project(PixelStretchTOP LANGUAGES CXX CUDA)

set(TD_SDK_DIR
    "C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus/CudaTOP"
    CACHE PATH "Folder containing TOP_CPlusPlusBase.h and CPlusPlus_Common.h")

option(PS_ALLOW_UNSUPPORTED_COMPILER "Pass -allow-unsupported-compiler to nvcc" ON)

if(NOT EXISTS "${TD_SDK_DIR}/TOP_CPlusPlusBase.h")
    message(FATAL_ERROR
        "TOP_CPlusPlusBase.h not found in TD_SDK_DIR='${TD_SDK_DIR}'.\n"
        "Point -DTD_SDK_DIR=... at a TD CPlusPlus sample folder (e.g. .../Samples/CPlusPlus/CudaTOP).")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Fat binary Turing..Blackwell + forward-compat PTX. Applied as a target property so it
# always wins over CMake's auto-default. Adjust for older toolkits (`nvcc --list-gpu-code`).
set(PS_CUDA_ARCHITECTURES "75-real;86-real;89-real;120-real;120-virtual"
    CACHE STRING "GPU architectures to embed (CMake CUDA_ARCHITECTURES syntax)")

find_package(CUDAToolkit REQUIRED)

add_library(PixelStretchTOP SHARED
    src/PixelStretchTOP.cpp
    src/PixelStretchCUDA.cu
)

target_include_directories(PixelStretchTOP PRIVATE
    "${TD_SDK_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${CUDAToolkit_INCLUDE_DIRS}
)
if(EXISTS "${CUDAToolkit_TARGET_DIR}/include/cccl")
    target_include_directories(PixelStretchTOP PRIVATE "${CUDAToolkit_TARGET_DIR}/include/cccl")
endif()

target_link_libraries(PixelStretchTOP PRIVATE CUDA::cudart_static)

set_target_properties(PixelStretchTOP PROPERTIES
    PREFIX ""
    OUTPUT_NAME "PixelStretchTOP"
    CUDA_ARCHITECTURES "${PS_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RUNTIME_LIBRARY Static
)

set_property(TARGET PixelStretchTOP PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

target_compile_definitions(PixelStretchTOP PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    WIN32_LEAN_AND_MEAN
)

target_compile_options(PixelStretchTOP PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
)
if(PS_ALLOW_UNSUPPORTED_COMPILER)
    target_compile_options(PixelStretchTOP PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)
endif()

# ---- Optional: auto-install to TD's per-user plugin folder -----------------
if(DEFINED ENV{USERPROFILE})
    set(TD_PLUGIN_DIR "$ENV{USERPROFILE}/Documents/Derivative/Plugins")
    add_custom_target(install_to_td
        COMMAND ${CMAKE_COMMAND} -E make_directory "${TD_PLUGIN_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:PixelStretchTOP>" "${TD_PLUGIN_DIR}/"
        DEPENDS PixelStretchTOP
        COMMENT "Copying PixelStretchTOP.dll to ${TD_PLUGIN_DIR}")
endif()
