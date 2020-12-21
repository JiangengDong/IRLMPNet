#ifndef IRLMPNET_COLLISION_CAR_H_
#define IRLMPNET_COLLISION_CAR_H_

#include <ompl/base/StateValidityChecker.h>
#include <boost/format.hpp>
#include <fstream>
#include <string>

namespace ob = ompl::base;

namespace IRLMPNet {
class DifferentialCarChecker : public ob::StateValidityChecker {
public:
    std::vector<std::array<double, 2>> obstacle_centers_;
    std::vector<std::array<double, 4>> obstacle_AABBs_;    
    double width_x_;
    double width_y_;
    double width_obs_;
    
    DifferentialCarChecker(ob::SpaceInformationPtr si, const unsigned char obs_index) : ob::StateValidityChecker(si) {
        // metrics
        width_x_ = 2.0;
        width_y_ = 1.0;
        width_obs_ = 8.0;
        _half_x = width_x_/2.0;
        _half_y = width_y_/2.0;
        _half_obs = width_obs_/2.0;
        _car_radius = sqrt(_half_x*_half_x + _half_y*_half_y);

        // read obstacle centers
        std::string obs_file = boost::str(boost::format("data/obstacle/car_obs/csv/obs_%d.csv") % obs_index);
        std::ifstream input_csv(obs_file);

        obstacle_centers_.resize(0);
        obstacle_AABBs_.resize(0);
        _obstacle_expanded_AABBs.resize(0);
        for(size_t i=0; i<5; i++) {
            std::string value_str;
            double center_x, center_y;

            getline(input_csv, value_str, ',');
            center_x = std::stod(value_str);
            getline(input_csv, value_str);
            center_y = std::stod(value_str);

            obstacle_centers_.emplace_back(center_x, center_y);
            obstacle_AABBs_.emplace_back(center_x - _half_obs, center_y - _half_obs, center_x + _half_obs, center_y + _half_obs);
            _obstacle_expanded_AABBs.emplace_back(
                center_x - _half_obs - _car_radius, 
                center_y - _half_obs - _car_radius, 
                center_x + _half_obs + _car_radius, 
                center_y + _half_obs + _car_radius);
        }

        input_csv.close();
    }

    bool isValid(const ob::State *state) const override {
        return true;
    }
private:
    std::vector<std::array<double, 4>> _obstacle_expanded_AABBs;
    double _half_x;
    double _half_y;
    double _half_obs;
    double _car_radius;

};
} // namespace IRLMPNet

#endif