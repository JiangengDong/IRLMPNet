#ifndef IRLMPNET_PLANNER_MPNETSAMPLER_H_
#define IRLMPNET_PLANNER_MPNETSAMPLER_H_

#include "planner/torch_interface/converter.h"
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <torch/script.h>
#include <fstream>

namespace ob = ompl::base;

namespace IRLMPNet {
class MPNetSampler : public ob::RealVectorStateSampler {
public:
    typedef std::shared_ptr<MPNetSampler> Ptr;

    torch::jit::Module mpnet_model_;
    unsigned int dim_;
    torch::Tensor voxel_;

    MPNetSampler(const ob::StateSpace *space, const std::string &mpnet_path) : ob::RealVectorStateSampler(space) {
        mpnet_model_ = torch::jit::load(mpnet_path);
        mpnet_model_.to(at::kCUDA);

        dim_ = space->as<ob::RealVectorStateSpace>()->getDimension();
        voxel_ = loadVoxel("data/voxel/car/voxel_0.csv").to(at::kCUDA);
    }

    void sampleMPNet(const ob::State *start, const ob::State *goal, ob::State *sample) {
        std::vector<double> start_vec, goal_vec;
        space_->copyToReals(start_vec, start);
        space_->copyToReals(goal_vec, goal);
        normalize(start_vec);
        normalize(goal_vec);

        auto mpnet_input = torch::cat({toTensor(start_vec, dim_), toTensor(goal_vec, dim_)}, 1).to(at::kCUDA);
        auto mpnet_output = mpnet_model_.forward({mpnet_input, voxel_}).toTensor().to(at::kCPU);

        // TODO: add normalize here
        auto sample_vec = toVector(mpnet_output, dim_);
        unnormalize(sample_vec);
        space_->copyFromReals(sample, sample_vec);
    }

    static torch::Tensor loadVoxel(const std::string &voxel_path) {
        unsigned int n_row = 0;
        std::vector<float> values;
        float val;

        std::ifstream csv_file(voxel_path);
        std::string line;

        while(std::getline(csv_file, line)) {
            std::stringstream ss(line);
            while (ss >> val) {
                values.emplace_back(val);
                if (ss.peek() == ',') {
                    ss.ignore();
                }
            }
            n_row += 1;
        }

        return torch::from_blob(values.data(), {1, 1, n_row, static_cast<unsigned int>(values.size())/n_row}).clone();
    }

    static void normalize(std::vector<double> &state) {
        state[0] /= 25;
        state[1] /= 35;
        state[2] /= M_PI;
    }

    static void unnormalize(std::vector<double> &state) {
        state[0] *= 25;
        state[1] *= 35;
        state[2] *= M_PI;
    }
};
} // namespace IRLMPNet

#endif