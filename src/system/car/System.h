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

namespace ob = ompl::base;
namespace oc = ompl::control;

namespace IRLMPNet {
    namespace System {
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

            // python interfaces below

            // state space related interface

            unsigned int getStateDim_py() const {
                return 3;
            }

            Eigen::VectorXf getStateLowerBound_py() const {
                const auto &bound = state_space->as<StateSpace>()->getBounds();
                Eigen::VectorXf v(getStateDim_py());
                for (unsigned int i = 0; i < getStateDim_py(); i++) {
                    v[i] = bound.low[i];
                }
                return v;
            }

            Eigen::Vector3f getStateUpperBound_py() const {
                const auto &bound = state_space->as<StateSpace>()->getBounds();
                Eigen::VectorXf v(getStateDim_py());
                for (unsigned int i = 0; i < getStateDim_py(); i++) {
                    v[i] = bound.high[i];
                }
                return v;
            }

            bool isValidState_py(Eigen::Ref<const Eigen::VectorXf> state_eig) const {
                ob::ScopedState<StateSpace> state(state_space);
                for (unsigned int i = 0; i < getStateDim_py(); i++) {
                    state[i] = state_eig[i];
                }
                return state_space->satisfiesBounds(state.get()) && space_information->isValid(state.get());
            }

            bool sampleValidState_py(Eigen::Ref<Eigen::VectorXf> state_eig) const {
                ob::ScopedState<StateSpace> state(state_space);
                if (sampleValidState(state.get())) {
                    for (unsigned int i = 0; i < getStateDim_py(); i++) {
                        state_eig[i] = state[i];
                    }
                    return true;
                } else {
                    return false;
                }
            }

            void enforceStateBound_py(Eigen::Ref<Eigen::VectorXf> state_eig) const {
                auto state = state_space->allocState();
                auto pstate = state->as<StateSpace::StateType>()->values;
                for (unsigned int i = 0; i < getStateDim_py(); i++) {
                    pstate[i] = state_eig[i];
                }
                state_space->enforceBounds(state);
                for (unsigned int i = 0; i < getStateDim_py(); i++) {
                    state_eig[i] = pstate[i];
                }
                state_space->freeState(state);
            }

            /// assume the two states are valid
            Eigen::VectorXf diffStates_py(Eigen::Ref<const Eigen::VectorXf> state1_eig, Eigen::Ref<const Eigen::VectorXf> state2_eig) const {
                Eigen::VectorXf diff(getStateDim_py());
                diff[0] = state1_eig[0] - state2_eig[0];
                diff[1] = state1_eig[1] - state2_eig[1];
                diff[2] = state1_eig[2] - state2_eig[2];
                if (diff[2] > M_PI) {
                    diff[2] -= 2 * M_PI;
                } else if (diff[2] < -M_PI) {
                    diff[2] += 2 * M_PI;
                }
                return diff;
            }

            float distance_py(Eigen::Ref<const Eigen::VectorXf> state1_eig, Eigen::Ref<const Eigen::VectorXf> state2_eig) const {
                return diffStates_py(state1_eig, state2_eig).norm();
            }

            // control space related interface

            unsigned int getControlDim_py() const {
                return 2;
            }

            Eigen::VectorXf getControlLowerBound_py() const {
                const auto &bound = control_space->as<ControlSpace>()->getBounds();
                Eigen::VectorXf v(getControlDim_py());
                for (unsigned int i = 0; i < getControlDim_py(); i++) {
                    v[i] = bound.low[i];
                }
                return v;
            }

            Eigen::VectorXf getControlUpperBound_py() const {
                const auto &bound = control_space->as<ControlSpace>()->getBounds();
                Eigen::VectorXf v(getControlDim_py());
                for (unsigned int i = 0; i < getControlDim_py(); i++) {
                    v[i] = bound.high[i];
                }
                return v;
            }

            bool isValidControl_py(Eigen::Ref<const Eigen::VectorXf> control_eig) const {
                const auto &bound = control_space->as<ControlSpace>()->getBounds();
                for (unsigned int i = 0; i < getControlDim_py(); i++) {
                    if (control_eig[i] < bound.low[i] || control_eig[i] > bound.high[i]) {
                        return false;
                    }
                }
                return true;
            }

            bool sampleValidControl_py(Eigen::Ref<Eigen::VectorXf> control_eig) const {
                auto control = control_space->allocControl();
                control_space->allocControlSampler()->sample(control);
                auto pcontrol = control->as<ControlSpace::ControlType>()->values;
                for (unsigned int i = 0; i < getControlDim_py(); i++) {
                    control_eig[i] = pcontrol[i];
                }
                control_space->freeControl(control);
                return true;
            }

            void enforceControlBound_py(Eigen::Ref<Eigen::VectorXf> control_eig) const {
                const auto &bound = control_space->as<ControlSpace>()->getBounds();
                for (unsigned int i = 0; i < getControlDim_py(); i++) {
                    if (control_eig[i] < bound.low[i]) {
                        control_eig[i] = bound.low[i];
                    }
                    if (control_eig[i] > bound.high[i]) {
                        control_eig[i] = bound.high[i];
                    }
                }
            }

            // propagation related

            float getPropagationStepSize_py() const {
                return space_information->getPropagationStepSize();
            }

            void propagate_py(Eigen::Ref<const Eigen::VectorXf> state_eig, Eigen::Ref<const Eigen::VectorXf> control_eig, Eigen::Ref<Eigen::VectorXf> result_eig, double duration) const {
                auto state = state_space->allocState();
                auto result = state_space->allocState();
                auto control = control_space->allocControl();

                auto pstate = state->as<StateSpace::StateType>()->values;
                for (unsigned int i = 0; i < getStateDim_py(); i++) {
                    pstate[i] = state_eig[i];
                }

                auto pcontrol = control->as<ControlSpace::ControlType>()->values;
                for (unsigned int i = 0; i < getControlDim_py(); i++) {
                    pcontrol[i] = control_eig[i];
                }

                space_information->propagateWhileValid(state, control, static_cast<int>(duration / space_information->getPropagationStepSize()), result);

                auto presult = result->as<StateSpace::StateType>()->values;
                for (unsigned int i = 0; i < getStateDim_py(); i++) {
                    result_eig[i] = presult[i];
                }
                state_space->freeState(state);
                state_space->freeState(result);
                control_space->freeControl(control);
            }

            Eigen::MatrixXf getLocalMap_py(Eigen::Ref<const Eigen::VectorXf> state_eig) const {
                Eigen::MatrixXf local_map(64, 64);
                local_map.fill(0.0);
                const auto &obs_AABBs = collision_checker->getObstacleAABBs();
                const float dir_x = cos(state_eig[2]) * 0.5, dir_y = sin(state_eig[2]) * 0.5;
                const float center_x = state_eig[0], center_y = state_eig[1];
                float x1, y1, x2, y2;
                for (unsigned int i = 0; i < 64; i++) {
                    x1 = (i - 31.5) * dir_x + center_x;
                    y1 = (i - 31.5) * dir_y + center_y;
                    for (unsigned int j = 0; j < 64; j++) {
                        x2 = -(j - 31.5) * dir_y + x1;
                        y2 = (j - 31.5) * dir_x + y1;

                        for (const auto &obs_AABB : obs_AABBs) {
                            if (x2 > obs_AABB[0] && x2 < obs_AABB[2] && y2 > obs_AABB[1] && y2 < obs_AABB[3]) {
                                local_map(i, j) = 1.0;
                                break;
                            }
                        }
                    }
                }
                return local_map;
            }
        };
    } // namespace System
} // namespace IRLMPNet

#endif //IRLMPNET_SYSTEM_CAR_SYSTEM_H
