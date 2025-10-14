cmake_minimum_required(VERSION 3.10)
project(ai_cpp_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add_executable(ai_cpp_demo src/main.cpp)
add_executable(multi_linear_regression src/multi_linear_regression.cpp)

set(SOURCE_FILES
src/main.cpp
src/k_means_clusterer.cpp 
src/logistic_regression.cpp
src/csv_parser.cpp
src/data_preprocessor.cpp   # <--- NEW FILE
# Add your original regression files here, e.g.,
# src/linear_regression.cpp
# src/multi_linear_regression.cpp
)

add_executable(ai_cpp_demo ${SOURCE_FILES})

