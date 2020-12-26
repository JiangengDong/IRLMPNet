#ifndef IRLMPNET_PLANNER_POLICY_H_
#define IRLMPNET_PLANNER_POLICY_H_

#include <torch/script.h>
#include "planner/torch_interface/converter.h"

namespace IRLMPNet {
class Policy {
public:
    torch::jit::script::Module policy_model_;
    unsigned int state_dim_;
    unsigned int control_dim_;

    Policy(const std::string model_path, const unsigned int state_dim, const unsigned int control_dim) {
        policy_model_ = torch::jit::load(model_path);
        policy_model_.to(at::kCUDA);

        state_dim_ = state_dim;
        control_dim_ = control_dim;
    }

    std::vector<double> act(const std::vector<double> &state) {
        auto policy_input = toTensor(state, state_dim_).to(at::kCUDA);
        auto policy_output = policy_model_.forward({policy_input}).toTensor().to(at::kCPU);
        return toVector(policy_output, control_dim_);
    }
};
} // namespace IRLMPNet

#endif