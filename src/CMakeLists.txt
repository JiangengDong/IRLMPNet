add_library(System
    system/car/src/System.cpp
    system/car/src/StateValidityChecker.cpp
)
target_include_directories(System PUBLIC
    ./)
target_link_libraries(System PUBLIC
    Eigen3::Eigen
    Boost::boost
    ompl_interface
)
set_target_properties(System PROPERTIES
    POSITION_INDEPENDENT_CODE TRUE)

# target: run planner with pure C++
add_executable(IRLMPNet
    main.cpp
    planner/src/RLMPNet.cpp
    planner/src/RLMPNetTree.cpp)
target_link_libraries(IRLMPNet PRIVATE
    torch
    Eigen3::Eigen
    Boost::boost
    ompl_interface
    System
    cnpy-static)


# export dynamic system to Python, so that both ends share the same system.
pybind11_add_module(KinoDynSys
    python.cpp)
target_include_directories(KinoDynSys PRIVATE
    ./)
target_link_libraries(KinoDynSys PRIVATE
    System)
set_target_properties(KinoDynSys PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/envs/")


# check the parts of this code individually
add_executable(test
    test.cpp
    test/src/test_car1order.cpp
    planner/src/RRTConnectMPNet.cpp
    planner/src/RRTMPNet.cpp)
target_include_directories(test PRIVATE
    ./)
target_link_libraries(test PRIVATE
    torch
    Eigen3::Eigen
    Boost::boost
    ompl_interface
    System)
