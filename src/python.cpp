//
// Created by jiangeng on 12/31/20.
//

#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "system/car/System.h"

namespace py = pybind11;

PYBIND11_MODULE(KinoDynSys, m) {
    py::class_<IRLMPNet::System::Car1OrderSystem>(m, "Car1OrderSystem").def(py::init<>());
}