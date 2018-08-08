
cmake_minimum_required(VERSION 3.6)
project(csim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

SET(BASE_DIR "csim")

find_package( OpenBLAS REQUIRED )

include_directories(${OpenBLAS_INCLUDE_DIRS})

add_executable(csim main.cc)

# link blas
target_link_libraries(csim
    ${OpenBLAS_LIBRARIES}
)