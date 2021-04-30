//
// Created by jiangeng on 1/1/21.
//

#include "planner/RRTConnectMPNet.h"
#include "planner/RRTMPNet.h"
#include "planner/torch_interface/MPNetSampler.h"
#include "planner/torch_interface/Policy.h"
#include "system/car/System.h"
#include <fstream>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>

namespace oc = ompl::control;
namespace ob = ompl::base;
namespace og = ompl::geometric;
using namespace IRLMPNet;

enum PlannerType {
    RRT = 0,
    RRTMPNet,
    RRTConnect,
    RRTConnectMPNet
};

void test_car1order_collision_checker() {
    auto system = IRLMPNet::System::Car1OrderSystem();
    auto sampler = system.space_information->allocStateSampler();
    auto rstate = system.space_information->allocState();

    std::ofstream output_csv("data/test/car1order_collision_states.csv");
    for (unsigned int i = 0; i < 100000; i++) {
        sampler->sampleUniform(rstate);
        if (!system.isValid(rstate)) {
            std::vector<double> state_vec;
            system.state_space->copyToReals(state_vec, rstate);
            for (unsigned int j = 0; j < state_vec.size(); j++) {
                output_csv << state_vec[j];
                if (j < state_vec.size() - 1) {
                    output_csv << ",";
                }
            }
            output_csv << std::endl;
        }
    }
    output_csv.close();
    system.space_information->freeState(rstate);
}

void test_car1order_propagator() {
    auto system = IRLMPNet::System::Car1OrderSystem();
    auto state_sampler = system.space_information->allocStateSampler();
    auto control_sampler = system.space_information->allocControlSampler();
    auto propagator = system.propagator;

    auto state = system.space_information->allocState();
    auto control = system.space_information->allocControl();

    state_sampler->sampleUniform(state);
    control_sampler->sample(control);

    std::ofstream output_csv("data/test/car1order_propagator_traj.csv");
    for (unsigned int i = 0; i < 100; i++) {
        std::vector<double> state_vec;
        system.state_space->copyToReals(state_vec, state);
        for (unsigned int j = 0; j < state_vec.size(); j++) {
            output_csv << state_vec[j];
            if (j < state_vec.size() - 1) {
                output_csv << ",";
            }
        }
        output_csv << std::endl;

        propagator->propagate(state, control, 0.1, state);
    }
    output_csv.close();
    system.space_information->freeControl(control);
    system.space_information->freeState(state);
}

void print_state(const ompl::base::StateSpacePtr &space, const ompl::base::State *state, std::ostream &o) {
    std::vector<double> state_temp;

    space->copyToReals(state_temp, state);
    for (unsigned int j = 0; j < state_temp.size(); j++) {
        o << state_temp[j];
        if (j < state_temp.size() - 1) {
            o << ",";
        }
    }
    o << std::endl;
}

void test_car1order_MPNet_unit() {
    auto system = IRLMPNet::System::Car1OrderSystem();
    auto mpnet_sampler = std::make_shared<IRLMPNet::MPNetSampler>(system.state_space.get(), "data/pytorch_model/mpnet/car/mpnet_script.pt");
    std::ofstream output_csv("data/test/car1order_mpnet_samples_unit.csv");
    std::vector<double> state_temp;

    std::vector<ompl::base::State *> start_states(5), goal_states(5);
    system.space_information->allocStates(start_states);
    system.space_information->allocStates(goal_states);
    auto sample_state = system.space_information->allocState();
    auto state_sampler = system.space_information->allocStateSampler();

    for (unsigned int i = 0; i < 5; i++) {
        state_sampler->sampleUniform(start_states[i]);
        print_state(system.state_space, start_states[i], output_csv);
    }

    for (unsigned int i = 0; i < 5; i++) {
        state_sampler->sampleUniform(goal_states[i]);
        print_state(system.state_space, goal_states[i], output_csv);
    }

    for (unsigned int i = 0; i < 5; i++) {
        for (unsigned int j = 0; j < 100; j++) {
            mpnet_sampler->sampleMPNet(start_states[i], goal_states[i], sample_state);
            print_state(system.state_space, sample_state, output_csv);
        }
    }

    output_csv.close();
    system.space_information->freeStates(start_states);
    system.space_information->freeStates(goal_states);
    system.space_information->freeState(sample_state);
}

void test_car1order_MPNet_integrate(
    std::vector<double> start_vec,
    std::vector<double> goal_vec,
    PlannerType planner_type,
    double planning_time) {
    auto car_system = std::make_shared<System::Car1OrderSystem>();
    auto simple_setup = std::make_shared<og::SimpleSetup>(car_system->space_information);
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
        case RRTConnect: {
            planner = std::make_shared<og::RRTConnect>(car_system->space_information);
            planner->as<og::RRTConnect>()->setRange(0.05);
            break;
        }
        case RRTConnectMPNet: {
            planner = std::make_shared<og::RRTConnectMPNet>(car_system->space_information);
            planner->as<og::RRTConnectMPNet>()->setRange(0.05);
            break;
        }
        case RRT: {
            planner = std::make_shared<og::RRT>(car_system->space_information);
            planner->as<og::RRT>()->setRange(0.05);
            planner->as<og::RRT>()->setGoalBias(0.05);
            break;
        }
        case RRTMPNet: {
            planner = std::make_shared<og::RRTMPNet>(car_system->space_information);
            planner->as<og::RRTMPNet>()->setRange(0.05);
            planner->as<og::RRTMPNet>()->setGoalBias(0.05);
            break;
        }
        default: {
            std::cout << "Invalid planner! " << std::endl;
            return;
        }
    }
    simple_setup->setPlanner(planner);
    simple_setup->setup();
    ob::PlannerStatus solved = simple_setup->solve(planning_time);

    if (solved == ob::PlannerStatus::EXACT_SOLUTION) {
        auto path = simple_setup->getSolutionPath();
        std::cout << "Found solution: "
                  << "path length = " << path.length() << std::endl;

        std::ofstream path_csv("./data/test/car1order_path.csv");
        path.printAsMatrix(path_csv);
        path_csv.close();

        switch (planner_type) {
            case RRTMPNet: {
                std::ofstream sample_csv("./data/test/car1order_mpnet_samples_integrate.csv");
                planner->as<og::RRTMPNet>()->printAllSamples(sample_csv);
                sample_csv.close();
                break;
            }
            case RRTConnectMPNet: {
                std::ofstream sample_csv("./data/test/car1order_mpnet_samples_integrate.csv");
                planner->as<og::RRTConnectMPNet>()->printAllSamples(sample_csv);
                sample_csv.close();
                break;
            }
            default: {
                break;
            }
        }
    } else {
        std::cout << "No solution found" << std::endl;
    }
}

void test_car1order_policy() {
    auto system = IRLMPNet::System::Car1OrderSystem();
    auto policy = IRLMPNet::Policy(system.space_information, "data/pytorch_model/rl/car_free-TD3-unnorm-script.pt");

    ompl::base::State *start = system.space_information->allocState(),
                      *goal = system.space_information->allocState(),
                      *next = system.space_information->allocState();
    ompl::control::Control *control = system.space_information->allocControl();

    system.state_space->copyFromReals(start, {-15, -10, 0});
    system.state_space->copyFromReals(goal, {0, 0, 0});

    std::ofstream output_csv("./data/test/car1order_policy_traj.csv");
    for (unsigned int i = 0; i < 300; i++) {
        policy.act(start, goal, control);
        system.space_information->propagate(start, control, 1, next);
        system.space_information->copyState(start, next);
        print_state(system.state_space, next, output_csv);
    }
}

void test_car1order() {
    // result: pass
    test_car1order_collision_checker();

    // // result: pass
    // test_car1order_propagator();

    // // result: pass
    // test_car1order_MPNet_unit();

    // // result: pass
    // test_car1order_MPNet_integrate({-20, -2, 0}, {-5, 30, 0}, RRTMPNet, 10.0);

    // // result: pass
    // test_car1order_policy();
}
