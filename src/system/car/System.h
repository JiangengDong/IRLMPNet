//
// Created by jiangeng on 12/31/20.
//

#ifndef IRLMPNET_SYSTEM_CAR_SYSTEM_H
#define IRLMPNET_SYSTEM_CAR_SYSTEM_H

#include <Eigen/Dense>
#include <ompl/base/ScopedState.h>
#include <ompl/control/SpaceInformation.h>

#include "system/car/ControlSpace.h"
#include "system/car/Propagator.h"
#include "system/car/StateSpace.h"
#include "system/car/StateValidityChecker.h"

namespace IRLMPNet {
    namespace System {
        namespace ob = ompl::base;
        namespace oc = ompl::control;

        class Car1OrderSystem {
        public:
            using StateSpace = Car1OrderStateSpace;
            using ControlSpace = Car1OrderControlSpace;
            using Propagator = Car1OrderPropagator;
            using CollisionChecker = CarCollisionChecker;

            ob::StateSpacePtr state_space;
            oc::ControlSpacePtr control_space;
            oc::StatePropagatorPtr propagator;
            CollisionChecker::Ptr collision_checker;
            oc::SpaceInformationPtr space_information;

            Car1OrderSystem(const unsigned int obstacle_index = 0) {
                state_space = std::make_shared<StateSpace>();
                control_space = std::make_shared<ControlSpace>(state_space);
                space_information = std::make_shared<oc::SpaceInformation>(state_space, control_space);
                propagator = std::make_shared<Propagator>(space_information);
                collision_checker = std::make_shared<CollisionChecker>(space_information, obstacle_index);
                space_information->setStatePropagator(propagator);
                space_information->setStateValidityChecker(collision_checker);
                space_information->setPropagationStepSize(0.002);
                space_information->setMinMaxControlDuration(100, 100);

                space_information->setup();
            }

            bool isValid(const ob::State *state) const {
                return space_information->isValid(state);
            }

            bool isValid(const std::vector<double> &state_vec) const {
                ob::ScopedState<StateSpace> state(state_space);
                state_space->copyFromReals(state.get(), state_vec);
                return isValid(state.get());
            }

            bool sampleValidState(ob::State *state) const {
                return space_information->allocValidStateSampler()->sample(state);
            }

            bool sampleValidState(std::vector<double> &state_vec) const {
                ob::ScopedState<StateSpace> state(state_space);
                if (sampleValidState(state.get())) {
                    state_space->copyToReals(state_vec, state.get());
                    return true;
                } else {
                    return false;
                }
            }
        };
    } // namespace System
} // namespace IRLMPNet

#endif //IRLMPNET_SYSTEM_CAR_SYSTEM_H
