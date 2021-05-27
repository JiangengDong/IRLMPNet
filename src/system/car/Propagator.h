//
// Created by jiangeng on 12/30/20.
//

#ifndef IRLMPNET_SYSTEM_CAR_PROPAGATOR_H
#define IRLMPNET_SYSTEM_CAR_PROPAGATOR_H

#include <algorithm>
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
            float integrate_stepsize_;

            explicit Car1OrderPropagator(const oc::SpaceInformationPtr &si) : oc::StatePropagator(si) {
                space_ = si_->getStateSpace();
                integrate_stepsize_ = 0.002;
            }

            void propagate(const ob::State *state, const oc::Control *control, double duration, ob::State *result) const override {
                space_->copyState(result, state);

                auto pstate = state->as<Car1OrderStateSpace::StateType>()->values;
                auto pcontrol = control->as<Car1OrderControlSpace::ControlType>()->values;
                auto presult = result->as<Car1OrderStateSpace::StateType>()->values;

                // convert to float
                float result_float[3], control_float[3];
                std::transform(pstate, pstate + 3, result_float, [](double x) -> float { return float(x); });
                std::transform(pcontrol, pcontrol + 2, control_float, [](double x) -> float { return float(x); });

                while (duration > 2 * integrate_stepsize_) {
                    RK4(result_float, 3, control_float, integrate_stepsize_, &ode, result_float);
                    std::transform(result_float, result_float + 3, presult, [](float x) -> double { return double(x); });
                    space_->enforceBounds(result);
                    std::transform(presult, presult + 3, result_float, [](double x) -> float { return float(x); });
                    duration -= integrate_stepsize_;
                }
                RK4(result_float, 3, control_float, float(duration), &ode, result_float);
                std::transform(result_float, result_float + 3, presult, [](float x) -> double { return double(x); });
                space_->enforceBounds(result);
                std::transform(presult, presult + 3, result_float, [](double x) -> float { return float(x); });
            }

            /// system dynamics
            static void ode(const float *state, const float *control, float *result) {
                const float &x = state[0], &y = state[1], &theta = state[2];
                const float &v = control[0], &w = control[1];

                result[0] = v * cos(theta);
                result[1] = v * sin(theta);
                result[2] = w;
            }
        };
    } // namespace System
} // namespace IRLMPNet

#endif //IRLMPNET_SYSTEM_CAR_PROPAGATOR_H
