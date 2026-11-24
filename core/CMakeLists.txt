cmake_minimum_required(VERSION 3.18)
project(livebeauty_core VERSION 3.6.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LIVEBEAUTY_BUILD_TESTS "Build C++ unit tests" ON)
option(LIVEBEAUTY_ENABLE_GLES "Build OpenGL ES 2.0/3.0 RHI backend" OFF)
option(LIVEBEAUTY_ENABLE_METAL "Build Apple Metal RHI backend" OFF)
option(LIVEBEAUTY_ENABLE_WEBGL "Build WebGL2 / Emscripten Wasm RHI backend" ${LIVEBEAUTY_BUILD_TESTS})
if(EMSCRIPTEN)
    set(LIVEBEAUTY_ENABLE_WEBGL ON CACHE BOOL "Build WebGL2 / Emscripten Wasm RHI backend" FORCE)
endif()
option(LIVEBEAUTY_BUILD_JNI "Build Android JNI bridge library" OFF)

# 1. Pure C++17 math, declarative .lbfx bundle parser & state-machine core (zero graphics/OS dependencies)
add_library(livebeauty_math STATIC
    src/config.cpp
    src/bundle/effect_bundle.cpp
    src/math/face_mesh_topology.cpp
    src/math/face_geometry.cpp
    src/math/face_roi_selector.cpp
    src/math/face_makeup_uv.cpp
    src/math/body_geometry.cpp
    src/math/beauty_tuning.cpp
    src/math/face_mesh_filter.cpp
    src/math/face_mesh_deformer.cpp
    src/math/face_timeline.cpp
    src/math/hair_mask_tuning.cpp
    src/math/sticker_transform.cpp
    src/math/multi_face_tracker.cpp
    src/math/face_relight.cpp
    src/math/device_tier.cpp
    src/math/ar_3d_accessory.cpp
    src/math/face_aux_vision.cpp
    src/math/face_parsing.cpp
    src/math/particle_system.cpp
)
target_include_directories(livebeauty_math PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(livebeauty_math PRIVATE
    $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wall -Wextra -Wno-unused-parameter>
)

# 2. Cross-backend RenderPipeline & LiveBeautyEngine facade (pure C++17)
add_library(livebeauty_engine STATIC
    src/rhi/render_pipeline.cpp
    src/engine.cpp
)
target_link_libraries(livebeauty_engine PUBLIC livebeauty_math)
target_compile_options(livebeauty_engine PRIVATE
    $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wall -Wextra -Wno-unused-parameter>
)

# 3. Optional OpenGL ES 2.0/3.0 RHI backend (Android NDK / GLES hosts)
if(LIVEBEAUTY_ENABLE_GLES)
    add_library(livebeauty_gles STATIC
        src/rhi/gles/gles_shaders.cpp
        src/rhi/gles/gles_device.cpp
    )
    target_include_directories(livebeauty_gles PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/rhi/gles>
    )
    target_link_libraries(livebeauty_gles PUBLIC livebeauty_engine GLESv2)
    target_compile_options(livebeauty_gles PRIVATE
        $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wall -Wextra -Wno-unused-parameter>
    )
endif()

# 4. Optional Android JNI shared library
if(LIVEBEAUTY_BUILD_JNI)
    if(NOT LIVEBEAUTY_ENABLE_GLES)
        message(FATAL_ERROR "LIVEBEAUTY_BUILD_JNI requires LIVEBEAUTY_ENABLE_GLES=ON")
    endif()
    add_library(livebeauty_jni SHARED
        src/jni/live_beauty_jni.cpp
    )
    target_link_libraries(livebeauty_jni PRIVATE livebeauty_gles log)
    target_compile_options(livebeauty_jni PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# 5. Optional Apple Metal RHI backend (iOS / macOS)
if(LIVEBEAUTY_ENABLE_METAL AND APPLE)
    enable_language(OBJCXX)
    add_library(livebeauty_metal STATIC
        src/rhi/metal/metal_device.mm
    )
    target_include_directories(livebeauty_metal PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/rhi/metal>
    )
    target_link_libraries(livebeauty_metal PUBLIC
        livebeauty_engine
        "-framework Foundation"
        "-framework Metal"
        "-framework CoreVideo"
        "-framework CoreGraphics"
    )
    target_compile_options(livebeauty_metal PRIVATE -fobjc-arc -Wall -Wextra -Wno-unused-parameter)
endif()

# 6. Optional WebGL2 / Emscripten WebAssembly RHI backend
if(LIVEBEAUTY_ENABLE_WEBGL)
    add_library(livebeauty_webgl STATIC
        src/rhi/gles/gles_shaders.cpp
        src/rhi/webgl/webgl2_device.cpp
    )
    target_link_libraries(livebeauty_webgl PUBLIC livebeauty_engine)
    target_compile_options(livebeauty_webgl PRIVATE
        $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wall -Wextra -Wno-unused-parameter>
    )
endif()

if(LIVEBEAUTY_BUILD_TESTS)
    enable_testing()
    add_executable(livebeauty_math_tests tests/test_math_core.cpp)
    target_link_libraries(livebeauty_math_tests PRIVATE livebeauty_math)
    add_test(NAME livebeauty_math_tests COMMAND livebeauty_math_tests)

    add_executable(livebeauty_rhi_tests tests/test_rhi_pipeline.cpp)
    if(LIVEBEAUTY_ENABLE_WEBGL)
        target_link_libraries(livebeauty_rhi_tests PRIVATE livebeauty_webgl)
        target_compile_definitions(livebeauty_rhi_tests PRIVATE LIVEBEAUTY_HAS_WEBGL_RHI=1)
    else()
        target_link_libraries(livebeauty_rhi_tests PRIVATE livebeauty_engine)
    endif()
    add_test(NAME livebeauty_rhi_tests COMMAND livebeauty_rhi_tests)

    add_executable(livebeauty_bench tests/bench_math_core.cpp)
    target_link_libraries(livebeauty_bench PRIVATE livebeauty_math)

    if(LIVEBEAUTY_ENABLE_METAL AND APPLE)
        add_executable(livebeauty_metal_pixel_gate tests/test_metal_pixel_gate.mm)
        target_link_libraries(livebeauty_metal_pixel_gate PRIVATE livebeauty_metal)
        target_compile_options(livebeauty_metal_pixel_gate PRIVATE -fobjc-arc)
        add_test(NAME livebeauty_metal_pixel_gate COMMAND livebeauty_metal_pixel_gate)
    endif()
endif()
