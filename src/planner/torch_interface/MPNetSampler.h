#ifndef IRLMPNET_PLANNER_MPNETSAMPLER_H_
#define IRLMPNET_PLANNER_MPNETSAMPLER_H_

#include "planner/torch_interface/converter.h"
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <torch/script.h>

namespace ob = ompl::base;

namespace IRLMPNet {
class MPNetSampler : public ob::RealVectorStateSampler {
public:
    typedef std::shared_ptr<MPNetSampler> Ptr;

    torch::jit::Module mpnet_model_;
    unsigned int dim_;

    MPNetSampler(const ob::StateSpace *space, const std::string &mpnet_path) : ob::RealVectorStateSampler(space) {
        mpnet_model_ = torch::jit::load(mpnet_path);
        mpnet_model_.to(at::kCUDA);

        dim_ = space->as<ob::RealVectorStateSpace>()->getDimension();
    }

    void sampleMPNet(const ob::State *start, const ob::State *goal, ob::State *sample) {
        std::vector<double> start_vec, goal_vec;
        space_->copyToReals(start_vec, start);
        space_->copyToReals(goal_vec, goal);

        auto mpnet_input = torch::cat({toTensor(start_vec, dim_), toTensor(goal_vec, dim_)}, 1).to(at::kCUDA);
        auto mpnet_output = mpnet_model_.forward({mpnet_input}).toTensor().to(at::kCPU);

        // TODO: add normalize here
        auto sample_vec = toVector(mpnet_output, dim_);
        space_->copyFromReals(sample, sample_vec);
    }
}
} // namespace IRLMPNet

#endif