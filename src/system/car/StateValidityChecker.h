//
// Created by jiangeng on 12/31/20.
//

#ifndef IRLMPNET_SYSTEM_CAR_STATEVALIDITYCHECKER_H
#define IRLMPNET_SYSTEM_CAR_STATEVALIDITYCHECKER_H

#include <boost/format.hpp>
#include <fstream>
#include <ompl/base/StateValidityChecker.h>
#include <string>

#include "system/car/StateSpace.h"

namespace ob = ompl::base;

namespace IRLMPNet {
    namespace System {
        class CarCollisionChecker : public ob::StateValidityChecker {
        public:
            using Ptr = std::shared_ptr<CarCollisionChecker>;

            std::vector<std::array<double, 2>> obs_centers_;
            double car_width_x_;
            double car_width_y_;
            double obs_width_;

            CarCollisionChecker(const ob::SpaceInformationPtr &si, const unsigned int obs_index) : ob::StateValidityChecker(si) {
                // metrics
                car_width_x_ = 2.0;
                car_width_y_ = 1.0;
                obs_width_ = 8.0;
                _car_half_x = car_width_x_ / 2.0;
                _car_half_y = car_width_y_ / 2.0;
                _obs_half_width = obs_width_ / 2.0;
                _car_radius = sqrt(_car_half_x * _car_half_x + _car_half_y * _car_half_y);

                obs_centers_ = loadObstacle(obs_index);

                _obs_AABBs.resize(0);
                _obs_expanded_AABBs.resize(0);
                for (const auto &obs_center : obs_centers_) {
                    double center_x = obs_center[0], center_y = obs_center[1];
                    _obs_AABBs.emplace_back(std::array<double, 4>{
                        center_x - _obs_half_width,
                        center_y - _obs_half_width,
                        center_x + _obs_half_width,
                        center_y + _obs_half_width});
                    _obs_expanded_AABBs.emplace_back(std::array<double, 4>{
                        center_x - _obs_half_width - _car_radius,
                        center_y - _obs_half_width - _car_radius,
                        center_x + _obs_half_width + _car_radius,
                        center_y + _obs_half_width + _car_radius});
                }
            }

            bool isValid(const ob::State *state) const override {
                // use RealVectorStateSpace here, since the first three elements of 1 order and 2 order car states are both x, y, theta
                const double *pstate = state->as<ob::RealVectorStateSpace::StateType>()->values;
                const double &car_x = pstate[0], &car_y = pstate[1], &car_theta = pstate[2];
                const double c_theta = cos(car_theta), s_theta = sin(car_theta);
                const double abs_c_theta = abs(c_theta), abs_s_theta = abs(s_theta);

                for (size_t i = 0; i < 5; i++) {
                    const auto &expanded_AABB = _obs_expanded_AABBs[i];
                    const auto &obs_AABB = _obs_AABBs[i];

                    // fast check: car inside AABB
                    if (car_x >= obs_AABB[0] and
                        car_y >= obs_AABB[1] and
                        car_x <= obs_AABB[2] and
                        car_y <= obs_AABB[3]) {
                        return false;
                    }

                    // fast check: car surely outside AABB
                    if (car_x < expanded_AABB[0] or
                        car_y < expanded_AABB[1] or
                        car_x > expanded_AABB[2] or
                        car_y > expanded_AABB[3]) {
                        continue;
                    }

                    // check: one edge of AABB is a separate line
                    const double car_x_halfspan = abs_c_theta * _car_half_x + abs_s_theta * _car_half_y,
                                 car_y_halfspan = abs_s_theta * _car_half_x + abs_c_theta * _car_half_y;
                    if (car_x - car_x_halfspan > obs_AABB[2] or
                        car_x + car_x_halfspan < obs_AABB[0] or
                        car_y - car_y_halfspan > obs_AABB[3] or
                        car_y + car_y_halfspan < obs_AABB[1]) {
                        continue;
                    }

                    // check: one edge of car is a separate line.
                    // method: rotate both car and obstacle by -theta, so that the car is aligned to the xy axis.
                    const double car_x_rotate = c_theta * car_x + s_theta * car_y,
                                 car_y_rotate = -s_theta * car_x + c_theta * car_y;
                    const auto &obs_center = obs_centers_[i];
                    const double obs_center_x_rotate = c_theta * obs_center[0] + s_theta * obs_center[1],
                                 obs_center_y_rotate = -s_theta * obs_center[0] + c_theta * obs_center[1];
                    const double obs_halfspan = _obs_half_width * (abs_c_theta + abs_s_theta);
                    if (obs_center_x_rotate - obs_halfspan > car_x_rotate + _car_half_x or
                        obs_center_x_rotate + obs_halfspan < car_x_rotate - _car_half_x or
                        obs_center_y_rotate - obs_halfspan > car_y_rotate + _car_half_y or
                        obs_center_y_rotate + obs_halfspan < car_y_rotate - _car_half_y) {
                        continue;
                    }

                    // no separate line found indicating overlapping
                    return false;
                }
                return true;
            }

            const std::vector<std::array<double, 4>>& getObstacleAABBs() const {
                return _obs_AABBs;
            }

        private:
            std::vector<std::array<double, 4>> _obs_AABBs;
            std::vector<std::array<double, 4>> _obs_expanded_AABBs;
            double _car_half_x;
            double _car_half_y;
            double _car_radius;
            double _obs_half_width;

            static std::vector<std::array<double, 2>> loadObstacle(unsigned int n);
        };
    } // namespace System
} // namespace IRLMPNet

#endif //IRLMPNET_SYSTEM_CAR_STATEVALIDITYCHECKER_H
