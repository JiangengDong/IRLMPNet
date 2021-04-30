#include "planner/RLMPNet.h"
#include "system/car/System.h"
#include "planner/RLMPNetTree.h"
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/planners/sst/SST.h>

namespace oc = ompl::control;
namespace ob = ompl::base;
using namespace IRLMPNet;

enum PlannerType {
    SST = 0,
    RLMPNet,
    RLMPNetTree,
};

std::tuple<double, double> car1order_control(
    const std::vector<double> start_vec,
    const std::vector<double> goal_vec,
    const double goal_radius,
    const PlannerType planner_type,
    const double time_limit,
    const double cost_limit,
    const std::string output_filename) {
    double length = std::numeric_limits<double>::infinity(), time = std::numeric_limits<double>::infinity();
    auto car_system = std::make_shared<System::Car1OrderSystem>();
    auto simple_setup = std::make_shared<oc::SimpleSetup>(car_system->space_information);
    auto objective = std::make_shared<ob::PathLengthOptimizationObjective>(car_system->space_information);
    objective->setCostThreshold(ob::Cost(cost_limit));
    simple_setup->setOptimizationObjective(objective);

    ob::ScopedState<System::Car1OrderStateSpace> start(car_system->state_space);
    car_system->state_space->copyFromReals(start.get(), start_vec);
    ob::ScopedState<System::Car1OrderStateSpace> goal(car_system->state_space);
    car_system->state_space->copyFromReals(goal.get(), goal_vec);
    simple_setup->setStartAndGoalStates(start, goal, goal_radius);

    ob::PlannerPtr planner;
    switch (planner_type) {
        case SST: {
            planner = std::make_shared<oc::SST>(car_system->space_information);
            break;
        }
        case RLMPNet: {
            auto policy = std::make_shared<Policy>(car_system->space_information,
                                                   "data/car1order/pytorch_model/rl/car_free-TD3-unnorm-script.pt");
            auto mpnet_sampler = std::make_shared<MPNetSampler>(car_system->state_space.get(),
                                                                "data/car1order/pytorch_model/mpnet/mpnet_script.pt");
            planner = std::make_shared<oc::RLMPNet>(car_system->space_information, policy, mpnet_sampler);
            break;
        }
        case RLMPNetTree: {
            auto policy = std::make_shared<Policy>(car_system->space_information,
                                                   "data/car1order/pytorch_model/rl/car_free-TD3-unnorm-script.pt");
            auto mpnet_sampler = std::make_shared<MPNetSampler>(car_system->state_space.get(),
                                                                "data/car1order/pytorch_model/mpnet/mpnet_script.pt");
            planner = std::make_shared<oc::RLMPNetTree>(car_system->space_information, policy, mpnet_sampler, 32);
            break;
        }
        default: {
            std::cout << "Invalid planner! " << std::endl;
            return {time, length};
        }
    }
    simple_setup->setPlanner(planner);
    simple_setup->setup();
    ob::PlannerStatus solved = simple_setup->solve(time_limit);

    if (solved == ob::PlannerStatus::EXACT_SOLUTION) {
        std::cout << "Found solution" << std::endl;
        length = simple_setup->getSolutionPath().length();
        time = simple_setup->getLastPlanComputationTime();

        std::ofstream output_csv(output_filename);
        simple_setup->getSolutionPath().printAsMatrix(output_csv);
        output_csv.close();
    } else if (solved == ob::PlannerStatus::APPROXIMATE_SOLUTION) {
        std::cout << "Found approximate solution" << std::endl;
    } else {
        std::cout << "No solution found" << std::endl;
    }

    return {time, length};
}

using StatesVec = std::vector<std::vector<double>>;

std::tuple<StatesVec, StatesVec> sampleStartGoal(unsigned int n) {
    auto car_system = std::make_shared<System::Car1OrderSystem>();
    std::vector<std::vector<double>> starts(n), goals(n);

    for (unsigned int i = 0; i < n; i++) {
        car_system->sampleValidState(starts[i]);
        car_system->sampleValidState(goals[i]);
    }

    return {starts, goals};
}

std::tuple<StatesVec, StatesVec> loadStartGoal(unsigned int n) {
    std::ifstream start_csv("./data/car1order/start_goal/testing_start.csv");
    std::ifstream goal_csv("./data/car1order/start_goal/testing_goal.csv");

    if (start_csv.good() and goal_csv.good()) {
        // TODO: we'd better use a fixed testing set.
    }
}

int main(int argc, char **argv) {
    auto car_system = std::make_shared<System::Car1OrderSystem>();
    std::vector<double> start, goal;
    double time, length;

    unsigned int i = 0;
    while(i < 500) {
        car_system->sampleValidState(start);
        car_system->sampleValidState(goal);

        std::stringstream ss;
        ss << "/home/jiangeng/Workspace/IRLMPNet/data/car1order/test_traj/path" << i << ".csv";

        std::tie(time, length) = car1order_control(start, goal, 0.5, SST, 60.0, 500.0, ss.str());
        if(time < 499.0) {
            std::cout << "Find traj. Save in " << ss.str() << std::endl;
            i++;
        }
    }
}
