cmake_minimum_required(VERSION 3.10)
project(ai_cpp_demo VERSION 1.0)

# C++ Standard Configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic") # Recommended flags for good practice

# 1. Define all C++ implementation files
# NOTE: All files listed here must *not* contain their own 'int main()' function, 
# except for the primary entry point (src/main.cpp).
set(ML_SOURCE_FILES
    src/main.cpp                 # <--- The single entry point (contains main())
    src/k_means_clusterer.cpp    # <--- Component implementation
    src/logistic_regression.cpp  # <--- Component implementation
    src/csv_parser.cpp           # <--- Component implementation
    src/data_preprocessor.cpp    # <--- Component implementation
    src/multi_linear_regression.cpp # <--- Component implementation
    # Add any other core .cpp files here
)

# 2. Define the main application executable
# This creates a single binary named 'ai_cpp_demo'
add_executable(ai_cpp_demo ${ML_SOURCE_FILES})

# 3. Define the public header search path
# This tells the compiler to look in the 'include' directory for headers
# (e.g., when you use #include "csv_parser.h")
target_include_directories(ai_cpp_demo
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 4. Optional: Install the executable (useful for a complete project)
install(TARGETS ai_cpp_demo DESTINATION bin)
