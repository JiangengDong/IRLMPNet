#ifndef IRLMPNET_COLLISION_CAR_H_
#define IRLMPNET_COLLISION_CAR_H_

#include <boost/format.hpp>
#include <fstream>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <string>

namespace ob = ompl::base;

namespace IRLMPNet {
class DifferentialCarChecker : public ob::StateValidityChecker {
public:
    std::vector<std::array<double, 2>> obstacle_centers_;
    double width_x_;
    double width_y_;
    double width_obs_;

    DifferentialCarChecker(ob::SpaceInformationPtr si, const unsigned int obs_index) : ob::StateValidityChecker(si) {
        // metrics
        width_x_ = 2.0;
        width_y_ = 1.0;
        width_obs_ = 8.0;
        _half_x = width_x_ / 2.0;
        _half_y = width_y_ / 2.0;
        _half_obs = width_obs_ / 2.0;
        _car_radius = sqrt(_half_x * _half_x + _half_y * _half_y);

        // read obstacle centers
        std::string obs_file = boost::str(boost::format("data/obstacle/car_obs/csv/obs_%1%.csv") % obs_index);
        std::ifstream input_csv(obs_file);

        obstacle_centers_.resize(0);
        _obstacle_AABBs.resize(0);
        _obstacle_expanded_AABBs.resize(0);
        for (size_t i = 0; i < 5; i++) {
            double center_x, center_y;
            std::string value_str;

            getline(input_csv, value_str, ',');
            center_x = std::stod(value_str);
            getline(input_csv, value_str);
            center_y = std::stod(value_str);

            obstacle_centers_.emplace_back(std::array<double, 2>{center_x, center_y});
            _obstacle_AABBs.emplace_back(std::array<double, 4>{center_x - _half_obs, center_y - _half_obs, center_x + _half_obs, center_y + _half_obs});
            _obstacle_expanded_AABBs.emplace_back(std::array<double, 4>{
                center_x - _half_obs - _car_radius,
                center_y - _half_obs - _car_radius,
                center_x + _half_obs + _car_radius,
                center_y + _half_obs + _car_radius});
        }

        input_csv.close();
    }

    bool isValid(const ob::State *state) const override {
        const double *pstate = state->as<ob::RealVectorStateSpace::StateType>()->values;
        const double &car_x = pstate[0], &car_y = pstate[1], &car_theta = pstate[2];
        const double c_theta = cos(car_theta), s_theta = sin(car_theta);
        const double abs_c_theta = abs(c_theta), abs_s_theta = abs(s_theta);

        for (size_t i = 0; i < 5; i++) {
            const auto &expanded_AABB = _obstacle_expanded_AABBs[i];
            const auto &obs_AABB = _obstacle_AABBs[i];

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
            const double car_x_halfspan = abs_c_theta * _half_x + abs_s_theta * _half_y,
                         car_y_halfspan = abs_s_theta * _half_x + abs_c_theta * _half_y;
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
            const auto &obs_center = obstacle_centers_[i];
            const double obs_center_x_rotate = c_theta * obs_center[0] + s_theta * obs_center[1],
                         obs_center_y_rotate = -s_theta * obs_center[0] + c_theta * obs_center[1];
            const double obs_halfspan = _half_obs * (abs_c_theta + abs_s_theta);
            if (obs_center_x_rotate - obs_halfspan > car_x_rotate + _half_x or
                obs_center_x_rotate + obs_halfspan < car_x_rotate - _half_x or
                obs_center_y_rotate - obs_halfspan > car_y_rotate + _half_y or
                obs_center_y_rotate + obs_halfspan < car_y_rotate - _half_y) {
                continue;
            }

            // no separate line found indicating overlapping
            return false;
        }
        return true;
    }

private:
    std::vector<std::array<double, 4>> _obstacle_AABBs;
    std::vector<std::array<double, 4>> _obstacle_expanded_AABBs;
    double _half_x;
    double _half_y;
    double _half_obs;
    double _car_radius;
};
} // namespace IRLMPNet

#endif