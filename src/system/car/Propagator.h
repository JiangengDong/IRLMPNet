//
// Created by jiangeng on 12/30/20.
//

#ifndef IRLMPNET_SYSTEM_CAR_PROPAGATOR_H
#define IRLMPNET_SYSTEM_CAR_PROPAGATOR_H

#include <ompl/control/SpaceInformation.h>
#include <ompl/control/StatePropagator.h>

#include "system/ODESolver.h"
#include "system/car/ControlSpace.h"
#include "system/car/StateSpace.h"


namespace oc = ompl::control;
namespace ob = ompl::base;

namespace IRLMPNet {
    namespace System {

        /// \brief we assume the state space be 3-dim RealVectorSpace, and the control space be 2-dim RealVectorSpace
        class Car1OrderPropagator : public oc::StatePropagator {
        public:
            using Ptr = std::shared_ptr<Car1OrderPropagator>;

            ob::StateSpacePtr space_;
            double integrate_stepsize_;

            explicit Car1OrderPropagator(const oc::SpaceInformationPtr &si) : oc::StatePropagator(si) {
                space_ = si_->getStateSpace();
                integrate_stepsize_ = 0.002;
            }

            void propagate(const ob::State *state, const oc::Control *control, double duration, ob::State *result) const override {
                space_->copyState(result, state);

                auto pstate = state->as<Car1OrderStateSpace::StateType>()->values;
                auto pcontrol = control->as<Car1OrderControlSpace::ControlType>()->values;
                auto presult = result->as<Car1OrderStateSpace::StateType>()->values;

                while (duration > 2 * integrate_stepsize_) {
                    RK4(presult, 3, pcontrol, integrate_stepsize_, &ode, presult);
                    space_->enforceBounds(result);
                    duration -= integrate_stepsize_;
                }
                RK4(presult, 3, pcontrol, duration, &ode, presult);
                space_->enforceBounds(result);
            }

            /// system dynamics
            static void ode(const double *state, const double *control, double *result) {
                const double &x = state[0], &y = state[1], &theta = state[2];
                const double &v = control[0], &w = control[1];

                result[0] = v * cos(theta);
                result[1] = v * sin(theta);
                result[2] = w;
            }
        };
    } // namespace System
} // namespace IRLMPNet

#endif //IRLMPNET_SYSTEM_CAR_PROPAGATOR_H
