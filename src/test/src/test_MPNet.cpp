//
// Created by jiangeng on 1/2/21.
//

#include "test/test_MPNet.h"
#include "planner/torch_interface/MPNetSampler.h"
#include "system/car/System.h"

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

void test_car1order_MPNet() {
    auto system = IRLMPNet::System::Car1OrderSystem();
    auto mpnet_sampler = std::make_shared<IRLMPNet::MPNetSampler>(system.state_space.get(), "data/pytorch_model/mpnet/car/mpnet_script.pt");
    std::ofstream output_csv("data/test/car1order_mpnet_samples.csv");
    std::vector<double> state_temp;

    std::vector<ompl::base::State *> start_states(5), goal_states(5);
    system.space_information->allocStates(start_states);
    system.space_information->allocStates(goal_states);
    auto sample_state = system.space_information->allocState(); // TODO: free these states
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

void test_MPNet() {
    test_car1order_MPNet();
}
