#include "planner/RLMPNet.h"
#include "planner/RLMPNetTree.h"
#include "system/car/ControlSpace.h"
#include "system/car/System.h"
#include <boost/format/format_fwd.hpp>
#include <cmath>
#include <cnpy/cnpy.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/planners/sst/SST.h>
#include <ompl/util/Console.h>
#include <torch/script.h>
#include <tuple>

namespace oc = ompl::control;
namespace ob = ompl::base;
using namespace IRLMPNet;

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

std::tuple<StatesVec, StatesVec> loadStartGoal(unsigned int n, std::string start_filename, std::string goal_filename) {
    std::ifstream start_file(start_filename);
    std::ifstream goal_file(goal_filename);
    bool file_status = start_file.good() and goal_file.good();
    start_file.close();
    goal_file.close();

    std::vector<std::vector<double>> starts, goals;
    bool regenerate_status = false;

    if (file_status) {
        auto start_data = cnpy::npy_load(start_filename);
        auto goal_data = cnpy::npy_load(goal_filename);
        auto car_system = std::make_shared<System::Car1OrderSystem>();
        if (start_data.shape[0] == n and start_data.shape[1] == 3 and goal_data.shape[0] == n and goal_data.shape[1] == 3) {
            auto pstarts = start_data.data<double>();
            auto pgoals = goal_data.data<double>();
            for (unsigned int i = 0; i < n; i++) {
                std::vector<double> start{pstarts[3 * i], pstarts[3 * i + 1], pstarts[3 * i + 2]};
                std::vector<double> goal{pgoals[3 * i], pgoals[3 * i + 1], pgoals[3 * i + 2]};
                if (car_system->isValid(start) and car_system->isValid(goal)) {
                    starts.emplace_back(start);
                    goals.emplace_back(goal);
                } else {
                    regenerate_status = true;
                    OMPL_WARN("Encounter invalid start or goal. Regenerate the dataset.");
                    break;
                }
            }
        } else {
            regenerate_status = true;
            OMPL_WARN("Dataset shape incorrect. Regenerate the dataset.");
        }
    }

    regenerate_status = !file_status or regenerate_status;

    if (regenerate_status) {
        std::tie(starts, goals) = sampleStartGoal(n);

        // flat starts and goals for save
        std::vector<double> flatten_starts, flatten_goals;
        for (auto start : starts) {
            flatten_starts.insert(flatten_starts.end(), start.begin(), start.end());
        }
        for (auto goal : goals) {
            flatten_goals.insert(flatten_goals.end(), goal.begin(), goal.end());
        }
        cnpy::npy_save(start_filename, flatten_starts.data(), {n, 3});
        cnpy::npy_save(goal_filename, flatten_goals.data(), {n, 3});
    }

    return {starts, goals};
}

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
    const std::string output_filename,
    const unsigned int index) {
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
    std::cout << "start: ";
    car_system->space_information->printState(start.get());
    std::cout << "goal:  ";
    car_system->space_information->printState(goal.get());

    ob::PlannerPtr planner;
    switch (planner_type) {
        case SST: {
            planner = std::make_shared<oc::SST>(car_system->space_information);
            break;
        }
        case RLMPNet: {
            auto planner_model = torch::jit::load("data/car1order/rl_result/test2/torchscript/mpc_planner.pth");
            auto transition_model = torch::jit::load("data/car1order/rl_result/test2/torchscript/transition_model.pth");
            auto encoder_model = torch::jit::load("data/car1order/rl_result/test2/torchscript/observation_encoder.pth");
            auto policy = std::make_shared<Policy>(car_system->space_information, planner_model, transition_model, encoder_model);
            auto mpnet_model = torch::jit::load("data/car1order/mpnet_result/default/torchscript/mpnet_script.pt");
            auto mpnet_sampler = std::make_shared<MPNetSampler>(car_system->state_space.get(), mpnet_model);
            planner = std::make_shared<oc::RLMPNet>(car_system->space_information, policy, mpnet_sampler);

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

        auto path = simple_setup->getSolutionPath();
        auto states = path.getStates();
        auto controls = path.getControls();
        auto durations = path.getControlDurations();
        const auto n = path.getStateCount();
        const size_t state_size = 3;
        const size_t control_size = 2;
        const size_t row_size = state_size + control_size + 1;
        std::vector<double> flatten_traj(n * row_size);
        // first n-1 elements
        for (size_t i = 0; i < n - 1; i++) {
            auto pstate = states[i]->as<System::Car1OrderStateSpace::StateType>()->values;
            auto pcontrol = controls[i]->as<System::Car1OrderControlSpace::ControlType>()->values;
            for (size_t j = 0; j < state_size; j++) {
                flatten_traj[i * row_size + j] = pstate[j];
            }
            for (size_t j = 0; j < control_size; j++) {
                flatten_traj[i * row_size + state_size + j] = pcontrol[j];
            }
            flatten_traj[i * row_size + state_size + control_size] = durations[i];
        }
        // last state has no following controls
        size_t i = n - 1;
        auto pstate = states[i]->as<System::Car1OrderStateSpace::StateType>()->values;
        for (size_t j = 0; j < state_size; j++) {
            flatten_traj[i * row_size + j] = pstate[j];
        }
        for (size_t j = 0; j < control_size; j++) {
            flatten_traj[i * row_size + state_size + j] = 0;
        }
        flatten_traj[i * row_size + state_size + control_size] = 0;

        // save file
        auto mode = (index == 0) ? "w" : "a";
        std::string dataset_name = (boost::format("traj%d") % index).str();
        cnpy::npz_save(output_filename, dataset_name, flatten_traj.data(), {n, row_size}, mode);
    } else if (solved == ob::PlannerStatus::APPROXIMATE_SOLUTION) {
        std::cout << "Found approximate solution" << std::endl;
    } else {
        std::cout << "No solution found" << std::endl;
    }

    return {time, length};
}

int main(int argc, char **argv) {
    const unsigned int N = 500;
    auto [starts, goals] = loadStartGoal(N, "./data/car1order/test_traj/test_starts.npy", "./data/car1order/test_traj/test_goals.npy");
    auto output_filename = "data/car1order/test_traj/test_traj.npz";

    for (unsigned int i = 0; i < N; i++) {
        auto [time, length] = car1order_control(starts[i], goals[i], 0.5, SST, 60.0, 500.0, output_filename, i);
        if (!std::isinf(time)) {
            std::cout << "Find traj " << i << std::endl;
        }
    }
}
