# target 1: run planner with pure C++
add_executable(IRLMPNet
        ./main.cpp
        planner/src/RLMPNet.cpp)
target_include_directories(IRLMPNet PRIVATE
        ./)
target_link_libraries(IRLMPNet PRIVATE
        torch
        Eigen3::Eigen
        Boost::boost
        ompl_interface)


# target 2: export dynamic system to Python, so that both ends share the same system.
pybind11_add_module(KinoDynSys
        ./python.cpp)
target_include_directories(KinoDynSys PRIVATE
        ./)
target_link_libraries(KinoDynSys PRIVATE
        Eigen3::Eigen
        ompl_interface)