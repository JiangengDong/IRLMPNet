#ifndef IRLMPNET_PLANNER_POLICY_H_
#define IRLMPNET_PLANNER_POLICY_H_

#include <torch/script.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/control/SpaceInformation.h>

#include "planner/torch_interface/converter.h"

namespace ob = ompl::base;
namespace oc = ompl::control;

namespace IRLMPNet {
    class Policy {
    public:
        torch::jit::script::Module policy_model_;
        unsigned int state_dim_;
        unsigned int control_dim_;
        ob::StateSpacePtr state_space_;
        oc::ControlSpacePtr control_space_;

        Policy(const oc::SpaceInformationPtr& space_information, const std::string& model_path) {
            policy_model_ = torch::jit::load(model_path);
            policy_model_.to(at::kCUDA);

            state_dim_ = space_information->getStateDimension();
            control_dim_ = space_information->getControlSpace()->getDimension();
            state_space_ = space_information->getStateSpace(); // Used for debug
            control_space_ = space_information->getControlSpace();
        }

        std::vector<double> act(const std::vector<double> &state) {
            auto policy_input = toTensor(state, state_dim_).to(at::kCUDA);
            auto policy_output = policy_model_.forward({policy_input}).toTensor().to(at::kCPU);
            return toVector(policy_output, control_dim_);
        }

        /// WARNING: The start, goal and control are assumed to be real vector
        void act(const ob::State* start, const ob::State* goal, oc::Control* control) {
            state_space_->printState(start, std::cout);
            state_space_->printState(goal, std::cout);
            // TODO: use profile to check if the copy here is optimized by the compiler
            std::vector<double> state_vec(2 * state_dim_);
            const auto* pstart = start->as<ob::RealVectorStateSpace::StateType>()->values;
            const auto* pgoal = goal->as<ob::RealVectorStateSpace::StateType>()->values;

            // normalize
            auto x_max = std::max(pstart[0], pgoal[0]);
            auto x_min = std::min(pstart[0], pgoal[0]);
            auto x_range = x_max - x_min;
            auto y_max = std::max(pstart[1], pgoal[1]);
            auto y_min = std::min(pstart[1], pgoal[1]);
            auto y_range = y_max - y_min;
            auto xy_range = std::max(x_range, y_range);

            state_vec[0] = 2*(pstart[0] - x_min)/xy_range - 1;
            state_vec[1] = 2*(pstart[1] - y_min)/xy_range - 1;
            state_vec[2] = pstart[2] / M_PI;
            state_vec[3] = 2*(pgoal[0] - x_min)/xy_range - 1;
            state_vec[4] = 2*(pgoal[1] - y_min)/xy_range - 1;
            state_vec[5] = pgoal[2] / M_PI;

//            for (unsigned int i=0; i<state_dim_; i++) {
//                state_vec[i] = pstart[i];
//                state_vec[i + state_dim_] = pgoal[i];
//            }
            std::cout << state_vec << std::endl;

            auto policy_input = toTensor(state_vec, state_dim_*2).to(at::kCUDA);
            auto policy_output = policy_model_.forward({policy_input}).toTensor().to(at::kCPU);

            auto control_vec = toVector(policy_output, control_dim_);
            auto* pcontrol = control->as<oc::RealVectorControlSpace::ControlType>()->values;
            for (unsigned int i=0; i<control_dim_; i++) {
                pcontrol[i] = control_vec[i];
            }
            control_space_->printControl(control, std::cout);
        }
    };
} // namespace IRLMPNet

#endif