cmake_minimum_required(VERSION 3.10)
project(ai_lab_demo VERSION 1.0)

# C++ standard configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

# Collect sources automatically
file(GLOB_RECURSE APP_SOURCES app/*.cpp)
file(GLOB_RECURSE DATA_SOURCES data/preprocessing/*.cpp)
file(GLOB_RECURSE MODEL_LINEAR_SOURCES models/linear/*.cpp)

file(GLOB_RECURSE MODEL_CLUSTER_SOURCES models/clustering/*.cpp)
file(GLOB_RECURSE MODEL_DECISION_TREE_SOURCES models/decision_tree/*.cpp)
file(GLOB_RECURSE MODEL_NEURAL_NETWORK_SOURCES models/neural/*.cpp)

# Create executable
add_executable(ai_lab_demo

    ${APP_SOURCES}
    ${DATA_SOURCES}
    ${MODEL_LINEAR_SOURCES}

    ${MODEL_CLUSTER_SOURCES}
    ${MODEL_DECISION_TREE_SOURCES}
    ${MODEL_NEURAL_NETWORK_SOURCES}
)

# Include directories
target_include_directories(ai_lab_demo
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optional install step
install(TARGETS ai_lab_demo DESTINATION bin)
