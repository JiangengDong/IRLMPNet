cmake_minimum_required (VERSION 3.5)
project(IRLMPNet)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

find_package(ompl 1.4 REQUIRED)
add_library(ompl_interface INTERFACE)
target_include_directories(ompl_interface
    INTERFACE ${OMPL_INCLUDE_DIRS})
target_link_libraries(ompl_interface
    INTERFACE ${OMPL_LIBRARIES})

find_package(Boost REQUIRED)

# WARNING: Don't use C++11 when using Torch > 1.4!
# WARNING: Build torch from source!
# Note: Add the location of torch to CMAKE_PREFIX_PATH.
#       You can use an environment variable `CMAKE_PREFIX_PATH`
#         or a command line argument `cmake .. -DCMAKE_PREFIX_PATH=<...>`.
#       The typical location is `<pytorch source>/torch/share/cmake`.
find_package(Torch REQUIRED)

add_subdirectory(extern/pybind11)

find_package(ZLIB)
add_library(cnpy-static STATIC
    extern/cnpy/cnpy.cpp)
target_link_libraries(cnpy-static PUBLIC
    ${ZLIB_LIBRARIES})
target_include_directories(cnpy-static PUBLIC
    ${ZLIB_INCLUDE_DIRS}
    ./extern)

add_subdirectory(src bin)
