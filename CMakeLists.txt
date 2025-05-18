cmake_minimum_required(VERSION 3.10)
project(DigitRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LIBSVM package
find_library(LIBSVM_LIBRARY svm REQUIRED)
find_path(LIBSVM_INCLUDE_DIR svm.h REQUIRED)

# Add source files
set(SOURCES
    src/main.cpp
    src/dataset.cpp
    src/svm.cpp
)

# Create executable
add_executable(digit_recognition ${SOURCES})

# Include directories
target_include_directories(digit_recognition PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${LIBSVM_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(digit_recognition PRIVATE ${LIBSVM_LIBRARY}) 