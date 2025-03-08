cmake_minimum_required(VERSION 3.15)
project(inference_engine CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)
find_package(nlohmann_json REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wpointer-arith -Werror -O3 -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -Wpointer-arith -Werror -O3 -ffast-math")

# Find all source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/*.c"
)

# Main executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    fmt::fmt
    nlohmann_json::nlohmann_json
)

# Include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)