cmake_minimum_required (VERSION 3.10)
project(IRLMPNet)

find_package(Eigen3 REQUIRED)

find_package(ompl REQUIRED)
add_library(ompl_interface INTERFACE)
target_link_libraries(ompl_interface
    INTERFACE ${OMPL_LIBRARIES})
target_include_directories(ompl_interface
    INTERFACE ${OMPL_INCLUDE_DIRS})

find_package(Boost REQUIRED)

find_package(Torch REQUIRED)

# add_subdirectory(extern/pybind11)

add_subdirectory(src bin)