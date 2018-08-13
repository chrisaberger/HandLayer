
cmake_minimum_required(VERSION 3.6)
project(csim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories("..")

add_executable(all_tests 
  all_tests.cc
  test_embedding.cc
  test_lstm.cc
  test_linear.cc
  test_softmax.cc)

# link blas
target_link_libraries(all_tests
    csim
)