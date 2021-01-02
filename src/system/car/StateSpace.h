//
// Created by jiangeng on 12/30/20.
//

#ifndef IRLMPNET_SYSTEM_CAR_STATESPACE_H
#define IRLMPNET_SYSTEM_CAR_STATESPACE_H

#include <ompl/base/spaces/RealVectorStateSpace.h>

namespace ob = ompl::base;

namespace IRLMPNet {
    namespace System {
        /// \brief My implementation for SE2 state spae.
        ///
        /// Unlike ompl::base::SE2StateSpace, the layout of the state is fixed, so I am confident to manipulate it with pointers.
        class Car1OrderStateSpace : public ob::RealVectorStateSpace {
        public:
            using Ptr = std::shared_ptr<Car1OrderStateSpace>;

            Car1OrderStateSpace() : ob::RealVectorStateSpace(3) {
                auto bounds = ob::RealVectorBounds(3);
                bounds.low = {-25, -35, -M_PI};
                bounds.high = {25, 35, M_PI};
                setBounds(bounds);

                setDimensionName(0, "x");
                setDimensionName(1, "y");
                setDimensionName(2, "theta");
            }

            void enforceBounds(ob::State *state) const override {
                auto pstate = state->as<StateType>()->values;
                pstate[2] = std::remainder(pstate[2], 2 * M_PI);
                ob::RealVectorStateSpace::enforceBounds(state);
            }

            double distance(const ob::State *state1, const ob::State *state2) const override {
                const auto *pstate1 = state1->as<StateType>()->values;
                const auto *pstate2 = state2->as<StateType>()->values;
                const double x_diff = pstate1[0] - pstate2[0], y_diff = pstate1[1] - pstate2[1];
                double theta_diff = pstate1[2] - pstate2[2];
                if (theta_diff > M_PI) {
                    theta_diff -= 2 * M_PI;
                }
                if (theta_diff < -M_PI) {
                    theta_diff += 2 * M_PI;
                }
                return std::sqrt(x_diff * x_diff + y_diff * y_diff + theta_diff * theta_diff);
            }

            bool equalStates(const ob::State *state1, const ob::State *state2) const override {
                const auto *pstate1 = state1->as<StateType>()->values;
                const auto *pstate2 = state2->as<StateType>()->values;
                double theta_diff = std::abs(pstate1[2] - pstate2[2]);
                if (theta_diff > M_PI) {
                    theta_diff = 2 * M_PI - theta_diff;
                }
                return pstate1[0] == pstate2[0] &&
                       pstate1[1] == pstate2[1] &&
                       theta_diff <= std::numeric_limits<double>::epsilon();
            }

            void interpolate(const ob::State *from, const ob::State *to, double t, ob::State *state) const override {
                const auto *pfrom = from->as<StateType>()->values;
                const auto *pto = to->as<StateType>()->values;
                auto *pstate = state->as<StateType>()->values;

                pstate[0] = pfrom[0] + (pto[0] - pfrom[0]) * t;
                pstate[1] = pfrom[1] + (pto[1] - pfrom[1]) * t;

                double theta_diff = pto[2] - pfrom[2];
                if (theta_diff > M_PI) {
                    theta_diff -= 2 * M_PI;
                }
                if (theta_diff < -M_PI) {
                    theta_diff += 2 * M_PI;
                }
                pstate[2] = pfrom[2] + theta_diff * t;
                if (pstate[2] > M_PI) {
                    pstate[2] -= 2 * M_PI;
                }
                if (pstate[2] < -M_PI) {
                    pstate[2] += 2 * M_PI;
                }
            }
        };
    }
}

#endif //IRLMPNET_SYSTEM_CAR_STATESPACE_H
