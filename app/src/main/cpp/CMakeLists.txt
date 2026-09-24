cmake_minimum_required(VERSION 3.22.1)
project(rvc_qnn LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(RVC_LIB_NAME rvc_qnn)

message(STATUS "QNN_SDK_DIR = ${QNN_SDK_DIR}")

include_directories(${QNN_SDK_DIR}/include/QNN)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_compile_options(
    -O2
    -fexceptions
    -frtti
    -Wall
    -Wno-unused-parameter
    -Wno-unused-variable
    -Wno-unused-function
)

# librvc_qnn.so — JNI bridge + QNN HTP executor.
# The QNN .so are NOT linked: they are dlopen'd at runtime from the app's own
# lib dir, exactly like qnn_resident.c did on the Termux side.
add_library(${RVC_LIB_NAME} SHARED
    rvc_jni.cpp
    qnn_executor.cpp
    dsp.cpp
    rvc_pipeline.cpp
    index_retrieval.cpp
)

find_library(log-lib log)

target_link_libraries(${RVC_LIB_NAME} PRIVATE
    ${log-lib}
    dl
    android
)
