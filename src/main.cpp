#include <ompl/control/planners/sst/SST.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include "planner/RLMPNet.h"
#include "system/car/System.h"

namespace oc = ompl::control;
namespace ob = ompl::base;
using namespace IRLMPNet;

enum PlannerType {
    SST = 0,
    RLMPNet
};

void car1order(std::vector<double> start_vec, std::vector<double> goal_vec, PlannerType planner_type, double planning_time) {
    auto car_system = std::make_shared<System::Car1OrderSystem>();
    auto simple_setup = std::make_shared<oc::SimpleSetup>(car_system->space_information);
    auto objective = std::make_shared<ob::PathLengthOptimizationObjective>(car_system->space_information);
    simple_setup->setOptimizationObjective(objective);

    ob::ScopedState<System::Car1OrderStateSpace> start(car_system->state_space);
    start[0] = start_vec[0];
    start[1] = start_vec[1];
    start[2] = start_vec[2];
    ob::ScopedState<System::Car1OrderStateSpace> goal(car_system->state_space);
    goal[0] = goal_vec[0];
    goal[1] = goal_vec[1];
    goal[2] = goal_vec[2];
    simple_setup->setStartAndGoalStates(start, goal, 0.05);

    ob::PlannerPtr planner;
    switch (planner_type) {
        case SST: {
            planner = std::make_shared<oc::SST>(car_system->space_information);
            break;
        }
        case RLMPNet: {
            planner = std::make_shared<oc::RLMPNet>(car_system->space_information);
            break;
        }
        default: {
            break;
        }
    }
    simple_setup->setPlanner(planner);
    simple_setup->setup();
    ob::PlannerStatus solved = simple_setup->solve(planning_time);

    if (solved == ob::PlannerStatus::EXACT_SOLUTION) {
        std::cout << "Found solution: ";

        auto path = simple_setup->getSolutionPath();
        std::cout << "path length = " << path.length() << std::endl;
    } else
        std::cout << "No solution found" << std::endl;
}

int main(int argc, char **argv) {
    car1order({-0.5, 0, 0}, {0, 0.5, 0}, RLMPNet, 10.0);
}