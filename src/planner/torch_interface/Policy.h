#ifndef IRLMPNET_PLANNER_POLICY_H_
#define IRLMPNET_PLANNER_POLICY_H_

#include <ATen/Functions.h>
#include <c10/core/DeviceType.h>
#include <chrono>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <torch/script.h>

#include "planner/torch_interface/converter.h"
#include "system/car/StateValidityChecker.h"

namespace ob = ompl::base;
namespace oc = ompl::control;

namespace IRLMPNet {
    class Policy {
    public:
        using Ptr = std::shared_ptr<Policy>;
        torch::jit::script::Module policy_model_;
        torch::jit::script::Module transition_model_;
        torch::jit::script::Module encoder_model_;
        unsigned int state_dim_;
        unsigned int control_dim_;
        ob::StateSpacePtr state_space_;
        oc::ControlSpacePtr control_space_;
        ob::StateValidityCheckerPtr collision_checker_;

        Policy(const oc::SpaceInformationPtr &space_information, const std::string &model_path) {
            policy_model_ = torch::jit::load(model_path);
            policy_model_.to(at::kCUDA);
            transition_model_ = torch::jit::load("data/car1order/rl_result/default/torchscript/transition_model.pth");
            transition_model_.to(at::kCUDA);
            encoder_model_ = torch::jit::load("data/car1order/rl_result/default/torchscript/observation_encoder.pth");
            encoder_model_.to(at::kCUDA);

            state_dim_ = space_information->getStateDimension();
            control_dim_ = space_information->getControlSpace()->getDimension();
            state_space_ = space_information->getStateSpace(); // Used for debug
            control_space_ = space_information->getControlSpace();
            collision_checker_ = space_information->getStateValidityChecker();
        }

        /// WARNING: The start, goal and control are assumed to be real vector
        void act(const ob::State *start, const ob::State *goal, oc::Control *control) {
            std::vector<double> state_vec(64 * 64 * 3 + 2 * state_dim_);
            const auto *pstart = start->as<ob::RealVectorStateSpace::StateType>()->values;
            const auto *pgoal = goal->as<ob::RealVectorStateSpace::StateType>()->values;

            // normalize
            auto map = static_cast<IRLMPNet::System::CarCollisionChecker *>(collision_checker_.get())->getLocalMap(start);
            for (unsigned int ch = 0; ch < 3; ch++)
                for (unsigned int row = 0; row < 64; row++)
                    for (unsigned int col = 0; col < 64; col++)
                        state_vec[64 * 64 * ch + 64 * row + col] = map(row, col);
            state_vec[64 * 64 * 3 + 0] = pstart[0] / 25.0;
            state_vec[64 * 64 * 3 + 1] = pstart[1] / 35.0;
            state_vec[64 * 64 * 3 + 2] = pstart[2] / M_PI;
            state_vec[64 * 64 * 3 + 3] = pgoal[0] / 25.0;
            state_vec[64 * 64 * 3 + 4] = pgoal[1] / 35.0;
            state_vec[64 * 64 * 3 + 5] = pgoal[2] / M_PI;

            torch::NoGradGuard no_grad;
            auto observation = toTensor(state_vec, 64 * 64 * 3 + state_dim_ * 2).to(at::kCUDA);
            auto encoded_observation = encoder_model_.forward({observation}).toTensor();
            auto belief = torch::zeros({1, 256}).to(at::kCUDA);
            auto posterior_state = torch::zeros({1, 32}).to(at::kCUDA);
            auto action = torch::zeros({1, 2}).to(at::kCUDA);
            auto transition_output = transition_model_.forward({posterior_state, action.unsqueeze(0), belief, encoded_observation.unsqueeze(0)}).toList();
            belief = transition_output.get(0).toTensor().squeeze(0);
            posterior_state = transition_output.get(4).toTensor().squeeze(0);

            // auto start_time = std::chrono::system_clock::now();
            auto policy_output = policy_model_.forward({belief, posterior_state}).toTensor().to(at::kCPU);
            // auto end_time = std::chrono::system_clock::now();
            // std::chrono::duration<double> time = end_time - start_time;
            // std::cout << "duration: " << time.count() << std::endl;

            auto control_vec = toVector(policy_output, control_dim_);

            auto *pcontrol = control->as<oc::RealVectorControlSpace::ControlType>()->values;
            // for (unsigned int i = 0; i < control_dim_; i++) {
            //     pcontrol[i] = control_vec[i];
            // }
            pcontrol[0] = control_vec[0] * 3 + 0.5;
            pcontrol[1] = control_vec[1] * 2;
        }

        void actBatch(const std::vector<ob::State *> &starts, const std::vector<ob::State *> goals, std::vector<oc::Control *> &controls, const unsigned int n) {
            std::vector<double> goal_vec;
            std::vector<torch::Tensor> goal_tensors;

            for (size_t i = 0; i < n; i++) {
                const auto goal = goals[i];
                state_space_->copyToReals(goal_vec, goal);
                goal_tensors.emplace_back(toTensor(goal_vec, state_dim_));
            }
            const auto goal_tensor = torch::cat(goal_tensors, 0).to(at::kCUDA);

            actBatch(starts, goal_tensor, controls, n);
        }

        /// WARNING: goal_tensor should be on GPU
        void actBatch(const std::vector<ob::State *> &starts, const torch::Tensor &goal_tensor, std::vector<oc::Control *> &controls, const unsigned int n) {
            std::vector<double> start_vec;
            std::vector<torch::Tensor> start_tensors;

            for (size_t i = 0; i < n; i++) {
                const auto start = starts[i];
                state_space_->copyToReals(start_vec, start);
                start_tensors.emplace_back(toTensor(start_vec, state_dim_));
            }
            const auto start_tensor = torch::cat(start_tensors, 0).to(at::kCUDA);

            actBatch(start_tensor, goal_tensor, controls, n);
        }

        /// WARNING: start_tensor and goal_tensor should be on GPU
        void actBatch(const torch::Tensor &start_tensor, const torch::Tensor &goal_tensor, std::vector<oc::Control *> &controls, const unsigned int n) {
            torch::NoGradGuard no_grad;
            auto max_tensor = torch::maximum(start_tensor, goal_tensor);
            auto min_tensor = torch::minimum(start_tensor, goal_tensor);
            auto half_range_tensor = (max_tensor - min_tensor) / 2.0;
            auto center_tensor = (max_tensor + min_tensor) / 2.0;
            auto max_half_range_tensor = torch::amax(torch::slice(half_range_tensor, 1, 0, 2), 1, true).squeeze_();
            center_tensor.index_put_({"...", 2}, 0);
            half_range_tensor.index_put_({"...", 0}, max_half_range_tensor);
            half_range_tensor.index_put_({"...", 1}, max_half_range_tensor);
            half_range_tensor.index_put_({"...", 2}, M_PI);

            auto start_tensor_normalized = (start_tensor - center_tensor) / half_range_tensor;
            auto goal_tensor_normalized = (goal_tensor - center_tensor) / half_range_tensor;

            auto policy_input = torch::cat({start_tensor_normalized, goal_tensor_normalized}, 1);
            auto policy_output = policy_model_.forward({policy_input}).toTensor().to(at::kCPU);

            for (unsigned int i = 0; i < n; i++) {
                auto pcontrol = controls[i]->as<oc::RealVectorControlSpace::ControlType>()->values;
                const auto control_vec = toVector(policy_output.slice(0, i, i + 1, 1), control_dim_);
                for (unsigned int j = 0; j < control_dim_; j++) {
                    pcontrol[j] = control_vec[j];
                }
            }
        }
    };
} // namespace IRLMPNet

#endif
