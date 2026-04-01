cmake_minimum_required(VERSION 3.21)
project(vncv_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ─────────────────────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────────────────────

option(VNCV_BUILD_PYTHON "Build Python extension module" ON)
option(VNCV_USE_SYSTEM_ORT "Use system-installed ONNX Runtime instead of fetching" OFF)
option(VNCV_BUILD_SDK_TEST "Build C++ SDK smoke test executable" ON)

# ─────────────────────────────────────────────────────────────────────────────
# Dependencies fetched at configure time
# ─────────────────────────────────────────────────────────────────────────────

include(FetchContent)

# -- pybind11 -----------------------------------------------------------------
if(VNCV_BUILD_PYTHON)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# -- nlohmann/json ------------------------------------------------------------
FetchContent_Declare(
    nlohmann_json
    URL      https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# -- Clipper2 -----------------------------------------------------------------
FetchContent_Declare(
    Clipper2
    GIT_REPOSITORY https://github.com/AngusJohnson/Clipper2.git
    GIT_TAG        Clipper2_1.4.0
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  CPP
)
FetchContent_MakeAvailable(Clipper2)

# -- ONNX Runtime -------------------------------------------------------------
if(NOT VNCV_USE_SYSTEM_ORT)
    # Select the right pre-built package for the current platform/arch
    if(WIN32)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64")
            set(ORT_PLATFORM "win-arm64")
        else()
            set(ORT_PLATFORM "win-x64")
        endif()
        set(ORT_ARCHIVE_EXT "zip")
    elseif(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|ARM64")
            set(ORT_PLATFORM "osx-arm64")
        else()
            set(ORT_PLATFORM "osx-x86_64")
        endif()
        set(ORT_ARCHIVE_EXT "tgz")
    else()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(ORT_PLATFORM "linux-aarch64")
        else()
            set(ORT_PLATFORM "linux-x64")
        endif()
        set(ORT_ARCHIVE_EXT "tgz")
    endif()

    set(ORT_VERSION "1.21.0")
    set(ORT_BASE_URL "https://github.com/microsoft/onnxruntime/releases/download")
    set(ORT_ARCHIVE_NAME "onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}.${ORT_ARCHIVE_EXT}")
    set(ORT_URL "${ORT_BASE_URL}/v${ORT_VERSION}/${ORT_ARCHIVE_NAME}")

    FetchContent_Declare(
        onnxruntime_prebuilt
        URL "${ORT_URL}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime_prebuilt)

    set(ORT_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}")
    set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
    set(ORT_LIB_DIR     "${ORT_ROOT}/lib")

    # Locate the shared library
    if(WIN32)
        find_library(ORT_LIBRARY onnxruntime
            PATHS "${ORT_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
        set(ORT_DLL "${ORT_ROOT}/lib/onnxruntime.dll")
    elseif(APPLE)
        find_library(ORT_LIBRARY onnxruntime
            PATHS "${ORT_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
    else()
        find_library(ORT_LIBRARY onnxruntime
            PATHS "${ORT_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
    endif()

    # Create an imported target
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ORT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
    )
    if(WIN32 AND ORT_DLL)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_IMPLIB "${ORT_LIBRARY}"
        )
    endif()
else()
    # Use system ORT (e.g. from conda or apt)
    find_package(onnxruntime REQUIRED)
    if(NOT TARGET onnxruntime)
        # Some distros provide it without a CMake config; fall back to find_path/find_library
        find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
            PATH_SUFFIXES onnxruntime)
        find_library(ORT_LIBRARY onnxruntime)
        add_library(onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ORT_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
        )
    endif()
endif()

# -- OpenCV -------------------------------------------------------------------
# Try normal CMake package discovery first.
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)

# Fallback for environments where OpenCV is installed in non-standard locations
# (common in CI images, Windows chocolatey packages, custom prefixes, etc.).
if(NOT OpenCV_FOUND)
    set(_opencv_hint_dirs
        "$ENV{OpenCV_DIR}"
        "$ENV{OPENCV_DIR}"
        "C:/tools/opencv/build/x64/vc16/lib"
        "C:/tools/opencv/build"
        "/usr/lib/x86_64-linux-gnu/cmake/opencv4"
        "/usr/lib/aarch64-linux-gnu/cmake/opencv4"
        "/usr/local/lib/cmake/opencv4"
        "/opt/homebrew/opt/opencv/lib/cmake/opencv4"
        "/usr/local/opt/opencv/lib/cmake/opencv4"
    )

    foreach(_opencv_dir IN LISTS _opencv_hint_dirs)
        if(_opencv_dir)
            find_package(OpenCV QUIET
                COMPONENTS core imgproc imgcodecs
                PATHS "${_opencv_dir}"
                NO_DEFAULT_PATH
            )
            if(OpenCV_FOUND)
                message(STATUS "Found OpenCV via hint: ${_opencv_dir}")
                break()
            endif()
        endif()
    endforeach()
endif()

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR
        "OpenCV not found. Set OpenCV_DIR/OPENCV_DIR to a directory containing OpenCVConfig.cmake.")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# C++ core static library
# ─────────────────────────────────────────────────────────────────────────────

add_library(vncv_core_lib STATIC
    csrc/utils.cpp
    csrc/vocab.cpp
    csrc/detection.cpp
    csrc/classification.cpp
    csrc/ctc_decoder.cpp
    csrc/recognition_en.cpp
    csrc/recognition_vi.cpp
    csrc/ocr_engine.cpp
)

target_include_directories(vncv_core_lib PUBLIC csrc)

target_link_libraries(vncv_core_lib PUBLIC
    onnxruntime
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
    Clipper2
)

# ─────────────────────────────────────────────────────────────────────────────
# Python extension module
# ─────────────────────────────────────────────────────────────────────────────

if(VNCV_BUILD_PYTHON)
    pybind11_add_module(_vncv_core MODULE csrc/bindings.cpp)
    target_link_libraries(_vncv_core PRIVATE vncv_core_lib)

    # Copy the ONNX Runtime shared library next to the Python extension so
    # it can be found at runtime without a system-wide installation.
    if(NOT VNCV_USE_SYSTEM_ORT)
        if(APPLE)
            # On macOS rpath is set relative to the module
            set_target_properties(_vncv_core PROPERTIES
                BUILD_RPATH   "@loader_path"
                INSTALL_RPATH "@loader_path"
            )
        elseif(NOT WIN32)
            set_target_properties(_vncv_core PROPERTIES
                BUILD_RPATH   "$ORIGIN"
                INSTALL_RPATH "$ORIGIN"
            )
        endif()

        # Install ONNX Runtime shared libraries alongside the Python module.
        # On macOS we need both the symlink and the versioned dylib so
        # delocate can resolve dependencies inside the wheel.
        if(APPLE)
            install(DIRECTORY "${ORT_LIB_DIR}/"
                    DESTINATION "${SKBUILD_PLATLIB_DIR}/vncv"
                    FILES_MATCHING
                    PATTERN "libonnxruntime*.dylib")
        elseif(WIN32)
            if(EXISTS "${ORT_DLL}")
                install(FILES "${ORT_DLL}"
                        DESTINATION "${SKBUILD_PLATLIB_DIR}/vncv")
            endif()
        else()
            install(DIRECTORY "${ORT_LIB_DIR}/"
                    DESTINATION "${SKBUILD_PLATLIB_DIR}/vncv"
                    FILES_MATCHING
                    PATTERN "libonnxruntime.so*")
        endif()
    endif()

    install(TARGETS _vncv_core
            DESTINATION "${SKBUILD_PLATLIB_DIR}/vncv")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Optional C++ SDK smoke test executable
# ─────────────────────────────────────────────────────────────────────────────
if(VNCV_BUILD_SDK_TEST)
    add_executable(vncv_sdk_test csrc/test_sdk.cpp)
    target_link_libraries(vncv_sdk_test PRIVATE vncv_core_lib)
endif()
