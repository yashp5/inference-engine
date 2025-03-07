# Create and enter build directory
mkdir -p build
cd build

# Install dependencies with Conan 2.x, explicitly setting output folder
conan install .. --output-folder=. --build=missing

# Configure with CMake
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build .