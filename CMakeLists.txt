
cmake_minimum_required(VERSION 3.6)
project(csim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

SET(BASE_DIR "csim")

find_package( OpenBLAS REQUIRED )
find_package( ZLIB REQUIRED )

include_directories(${OpenBLAS_INCLUDE_DIRS})
include_directories("src")
include_directories("third_party")

add_library(csim SHARED
    third_party/cnpy.cc
    src/tensor.cc
    src/layers/embedding.cc
    src/layers/linear.cc
    src/layers/lstm.cc
    src/layers/logsoftmax.cc
)

target_link_libraries(csim
    ${OpenBLAS_LIBRARIES}
    ${ZLIB_LIBRARIES}
)

file(GLOB files "apps/*.cc")
foreach(file ${files})
  get_filename_component(BASE_NAME ${file} NAME_WE)
  
  add_executable(${BASE_NAME} ${file})
  target_link_libraries(${BASE_NAME} csim)
endforeach()



add_subdirectory(test)

