#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/ODESolver.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/control/planners/sst/SST.h>

#include <propagator/Car.h>
#include <state_validity_checker/Car.h>

namespace oc = ompl::control;
namespace ob = ompl::base;

void plan1order() {
    auto state_space = std::make_shared<ob::RealVectorStateSpace>(3);
    auto state_bounds = ob::RealVectorBounds(3);
    state_bounds.low = {-25, -35, -M_PI};
    state_bounds.high = {25, 35, M_PI};
    state_space->setBounds(state_bounds);

    auto control_space = std::make_shared<oc::RealVectorControlSpace>(state_space, 2);
    auto control_bounds = ob::RealVectorBounds(2);
    control_bounds.low = {-1, -1};
    control_bounds.high = {1, 1};
    control_space->setBounds(control_bounds);

    auto space_information = std::make_shared<oc::SpaceInformation>(state_space, control_space);
    auto state_validity_checker = std::make_shared<ob::AllValidStateValidityChecker>(space_information);
    space_information->setStateValidityChecker(state_validity_checker);

    auto propagator = std::make_shared<IRLMPNet::DifferentialCar1OrderPropagator>(space_information);
    space_information->setStatePropagator(propagator);
    space_information->setPropagationStepSize(0.01);
    space_information->setMinMaxControlDuration(1, 10);

    ob::ScopedState<ob::RealVectorStateSpace> start(state_space);
    start->values[0] = -0.5;
    start->values[1] = 0;
    start->values[2] = 0;

    ob::ScopedState<ob::RealVectorStateSpace> goal(state_space);
    goal->values[0] = 0;
    goal->values[1] = 0.5;
    goal->values[2] = 0;

    oc::SimpleSetup simple_setup(space_information);
    simple_setup.setStartAndGoalStates(start, goal, 0.05);

    auto planner = std::make_shared<oc::SST>(space_information);
    simple_setup.setPlanner(planner);
    simple_setup.setup();

    ob::PlannerStatus solved = simple_setup.solve(100.0);

    if (solved) {
        std::cout << "Found solution:" << std::endl;

        simple_setup.getSolutionPath().asGeometric().printAsMatrix(std::cout);
    } else
        std::cout << "No solution found" << std::endl;
}

void plan2order() {
    auto state_space = std::make_shared<ob::RealVectorStateSpace>(5);
    auto state_bounds = ob::RealVectorBounds(5);
    state_bounds.low = {-10, -10, -M_PI, -1, -1};
    state_bounds.high = {10, 10, M_PI, 1, 1};
    state_space->setBounds(state_bounds);

    auto control_space = std::make_shared<oc::RealVectorControlSpace>(state_space, 2);
    auto control_bounds = ob::RealVectorBounds(2);
    control_bounds.low = {-1, -1};
    control_bounds.high = {1, 1};
    control_space->setBounds(control_bounds);

    auto space_information = std::make_shared<oc::SpaceInformation>(state_space, control_space);
    auto state_validity_checker = std::make_shared<ob::AllValidStateValidityChecker>(space_information);
    space_information->setStateValidityChecker(state_validity_checker);

    auto propagator = std::make_shared<IRLMPNet::DifferentialCar2OrderPropagator>(space_information);
    space_information->setStatePropagator(propagator);
    space_information->setPropagationStepSize(0.01);
    space_information->setMinMaxControlDuration(10, 10);

    ob::ScopedState<ob::RealVectorStateSpace> start(state_space);
    start->values[0] = -0.5;
    start->values[1] = 0;
    start->values[2] = 0;
    start->values[3] = 0;
    start->values[4] = 0;

    ob::ScopedState<ob::RealVectorStateSpace> goal(state_space);
    goal->values[0] = 0;
    goal->values[1] = 0.5;
    goal->values[2] = 0;
    goal->values[3] = 0;
    goal->values[4] = 0;

    oc::SimpleSetup simple_setup(space_information);
    simple_setup.setStartAndGoalStates(start, goal, 0.05);

    auto planner = std::make_shared<oc::SST>(space_information);
    simple_setup.setPlanner(planner);
    simple_setup.setup();

    ob::PlannerStatus solved = simple_setup.solve(100.0);

    if (solved) {
        std::cout << "Found solution:" << std::endl;

        simple_setup.getSolutionPath().asGeometric().printAsMatrix(std::cout);
    } else
        std::cout << "No solution found" << std::endl;
}

int main(int argc, char **argv) {
    plan1order();
}