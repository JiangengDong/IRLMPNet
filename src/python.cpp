//
// Created by jiangeng on 12/31/20.
//

#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "system/car/System.h"

namespace py = pybind11;
using Car1OrderSystem = IRLMPNet::System::Car1OrderSystem;

PYBIND11_MODULE(KinoDynSys, m) {
    py::class_<Car1OrderSystem>(m, "Car1OrderSystem")
        .def(py::init<const unsigned int>(), py::arg("obstacle_index")=0)
        .def_property_readonly("state_dim", &Car1OrderSystem::getStateDim_py)
        .def_property_readonly("state_lower_bound", &Car1OrderSystem::getStateLowerBound_py)
        .def_property_readonly("state_upper_bound", &Car1OrderSystem::getStateUpperBound_py)
        .def("is_valid_state", &Car1OrderSystem::isValidState_py)
        .def("sample_valid_state", &Car1OrderSystem::sampleValidState_py)
        .def("enforce_state_bound", &Car1OrderSystem::enforceStateBound_py)
        .def("diff", &Car1OrderSystem::diffStates_py)
        .def("distance", &Car1OrderSystem::distance_py)
        .def_property_readonly("control_dim", &Car1OrderSystem::getControlDim_py)
        .def_property_readonly("control_lower_bound", &Car1OrderSystem::getControlLowerBound_py)
        .def_property_readonly("control_upper_bound", &Car1OrderSystem::getControlUpperBound_py)
        .def("is_valid_control", &Car1OrderSystem::isValidControl_py)
        .def("sample_valid_control", &Car1OrderSystem::sampleValidControl_py)
        .def("enforce_control_bound", &Car1OrderSystem::enforceControlBound_py)
        .def_property_readonly("step_size", &Car1OrderSystem::getPropagationStepSize_py)
        .def("propagate", &Car1OrderSystem::propagate_py)
        .def("get_local_map", &Car1OrderSystem::getLocalMap_py);
}
