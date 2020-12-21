add_executable(IRLMPNet
    ./main.cpp
)

target_include_directories(IRLMPNet PRIVATE 
    ./
)

target_link_libraries(IRLMPNet PRIVATE
    Eigen3::Eigen
    ompl_interface
    Boost::boost
)