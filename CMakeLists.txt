cmake_minimum_required(VERSION 3.10)
project(DigitRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_library(LIBSVM_LIBRARY svm REQUIRED)
find_path(LIBSVM_INCLUDE_DIR svm.h REQUIRED)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Enable testing
enable_testing()

# Download and build GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# Add source files
set(SOURCES
    src/main.cpp
    src/dataset.cpp
    src/svm.cpp
    src/utils.cpp
    src/cli.cpp
    src/preprocessing.cpp
    src/preprocessing_advanced.cpp
)

# Create main executable
add_executable(digit_recognition ${SOURCES})

# Include directories
target_include_directories(digit_recognition PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${LIBSVM_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(digit_recognition PRIVATE 
    ${LIBSVM_LIBRARY}
    ${OpenCV_LIBS}
)

# Add compiler flags
if(MSVC)
    target_compile_options(digit_recognition PRIVATE /W4)
else()
    target_compile_options(digit_recognition PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Create test executables
set(TEST_SOURCES
    tests/test_preprocessing.cpp
    tests/test_preprocessing_advanced.cpp
)

set(INTEGRATION_TEST_SOURCES
    tests/test_preprocessing_integration.cpp
    src/preprocessing.cpp
    src/svm.cpp
    src/dataset.cpp
    src/utils.cpp
)

add_executable(unit_tests ${TEST_SOURCES})
add_executable(integration_tests ${INTEGRATION_TEST_SOURCES})

target_include_directories(unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${LIBSVM_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(integration_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${LIBSVM_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(unit_tests PRIVATE
    GTest::gtest_main
    ${LIBSVM_LIBRARY}
    ${OpenCV_LIBS}
)

target_link_libraries(integration_tests PRIVATE
    GTest::gtest_main
    ${LIBSVM_LIBRARY}
    ${OpenCV_LIBS}
)

# Register tests
include(GoogleTest)
gtest_discover_tests(unit_tests)
gtest_discover_tests(integration_tests)

# Install rules
install(TARGETS digit_recognition DESTINATION bin)
install(FILES 
    scripts/download_mnist.sh
    scripts/preprocess_examples.sh
    DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

# Add header files
set(HEADERS
    include/preprocessing.h
)

# Create library
add_library(preprocessing STATIC ${SOURCES} ${HEADERS})
target_include_directories(preprocessing PUBLIC include)

# Add tests
set(TEST_SOURCES
    tests/test_preprocessing.cpp
    tests/test_preprocessing_advanced.cpp
)

# Create test executable
add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests preprocessing ${GTEST_LIBRARIES} pthread)

# Enable testing
enable_testing()
add_test(NAME run_tests COMMAND run_tests) 