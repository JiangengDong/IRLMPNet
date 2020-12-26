add_executable(IRLMPNet
    ./main.cpp
    ./planner/src/SST.cpp
    ./planner/src/SSTRL.cpp
)

target_include_directories(IRLMPNet PRIVATE 
    ./
)

target_link_libraries(IRLMPNet PRIVATE
    Eigen3::Eigen
    ompl_interface
    Boost::boost
    torch
)