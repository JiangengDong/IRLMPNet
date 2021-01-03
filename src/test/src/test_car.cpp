//
// Created by jiangeng on 1/1/21.
//

#include "system/car/System.h"
#include <fstream>

void test_car1order_collision_checker() {
    auto system = IRLMPNet::System::Car1OrderSystem();
    auto sampler = system.space_information->allocStateSampler();
    auto rstate = system.space_information->allocState(); // TODO: free this state

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
}

void test_car1order_propagator() {
    auto system = IRLMPNet::System::Car1OrderSystem();
    auto state_sampler = system.space_information->allocStateSampler();
    auto control_sampler = system.space_information->allocControlSampler();
    auto propagator = system.propagator;

    auto state = system.space_information->allocState();
    auto control = system.space_information->allocControl(); // TODO: free the two states

    state_sampler->sampleUniform(state);
    control_sampler->sample(control);

    std::ofstream output_csv("data/test/car1order_propagator_traj.csv");
    for(unsigned int i=0; i<100; i++) {
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
}

void test_car1order() {
//    // result: pass
//    test_car1order_collision_checker();

//    // result: pass
//    test_car1order_propagator();
}

void test_car() {
    test_car1order();
}