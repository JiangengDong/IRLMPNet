#ifndef IRLMPNET_PLANNER_MPNETSAMPLER_H_
#define IRLMPNET_PLANNER_MPNETSAMPLER_H_

#include "planner/torch_interface/converter.h"
#include <fstream>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <torch/script.h>

namespace ob = ompl::base;

namespace IRLMPNet {
    class MPNetSampler : public ob::RealVectorStateSampler {
    public:
        typedef std::shared_ptr<MPNetSampler> Ptr;

        torch::jit::Module mpnet_model_;
        unsigned int dim_;
        torch::Tensor voxel_;
        torch::Tensor input_normalize_bias_;
        torch::Tensor input_normalize_scale_;
        torch::Tensor output_normalize_bias_;
        torch::Tensor output_normalize_scale_;

        MPNetSampler(const ob::StateSpace *space, const std::string &mpnet_path) : ob::RealVectorStateSampler(space) {
            mpnet_model_ = torch::jit::load(mpnet_path);
            mpnet_model_.to(at::kCUDA);

            dim_ = space->as<ob::RealVectorStateSpace>()->getDimension();
            voxel_ = loadVoxel("data/voxel/car/voxel_0.csv").to(at::kCUDA); //  TODO: turn the file path into an argument

            auto bounds = space->as<ob::RealVectorStateSpace>()->getBounds();
            torch::NoGradGuard no_grad;
            auto high_bound_tensor = toTensor(bounds.high, dim_);
            auto low_bound_tensor = toTensor(bounds.low, dim_);
            input_normalize_bias_ = torch::tile((high_bound_tensor + low_bound_tensor) / 2.0, {1, 2}).to(at::kCUDA);
            input_normalize_scale_ = torch::tile(2.0 / (high_bound_tensor - low_bound_tensor), {1, 2}).to(at::kCUDA); // take reciprocal in advance to accelerate speed
            output_normalize_bias_ = ((high_bound_tensor + low_bound_tensor) / 2.0).to(at::kCUDA);
            output_normalize_scale_ = ((high_bound_tensor - low_bound_tensor) / 2.0).to(at::kCUDA);
        }

        void sampleMPNet(const ob::State *start, const ob::State *goal, ob::State *sample) {
            std::vector<double> start_vec, goal_vec;
            space_->copyToReals(start_vec, start);
            space_->copyToReals(goal_vec, goal);

            torch::NoGradGuard no_grad;
            auto mpnet_input = torch::cat({toTensor(start_vec, dim_), toTensor(goal_vec, dim_)}, 1).to(at::kCUDA);
            auto mpnet_input_normalized = (mpnet_input - input_normalize_bias_) * input_normalize_scale_;
            auto mpnet_output_normalized = mpnet_model_.forward({mpnet_input_normalized, voxel_}).toTensor();
            auto mpnet_output = (mpnet_output_normalized * output_normalize_scale_ + output_normalize_bias_).to(at::kCPU);
            auto sample_vec = toVector(mpnet_output, dim_);

            space_->copyFromReals(sample, sample_vec);
        }

        torch::Tensor sampleMPNetBatch(const std::vector<ob::State *> &starts, const std::vector<ob::State *> &goals, std::vector<ob::State *> &samples, const unsigned int n) {
            std::vector<double> start_vec, goal_vec, sample_vec;
            std::vector<torch::Tensor> tensors;

            for (size_t i = 0; i < n; i++) {
                const auto &start = starts[i];
                const auto &goal = goals[i];
                space_->copyToReals(start_vec, start);
                space_->copyToReals(goal_vec, goal);
                tensors.emplace_back(torch::cat({toTensor(start_vec, dim_), toTensor(goal_vec, dim_)}, 1));
            }
            torch::NoGradGuard no_grad;
            auto mpnet_input = torch::cat(tensors, 0).to(at::kCUDA);
            auto mpnet_input_normalized = (mpnet_input - input_normalize_bias_) * input_normalize_scale_;
            auto mpnet_output_normalized = mpnet_model_.forward({mpnet_input_normalized,
                                                                 voxel_.broadcast_to({static_cast<signed int>(n), 1, 32, 32})})
                                               .toTensor(); // TODO: use a variable voxel size here
            auto mpnet_output = (mpnet_output_normalized * output_normalize_scale_ + output_normalize_bias_);

            auto mpnet_output_cpu = mpnet_output.to(at::kCPU);
            for (size_t i = 0; i < n; i++) {
                sample_vec = toVector(mpnet_output_cpu.slice(0, i, i + 1, 1), dim_);
                space_->copyFromReals(samples[i], sample_vec);
            }

            return mpnet_output;
        }

        static torch::Tensor loadVoxel(const std::string &voxel_path) {
            unsigned int n_row = 0;
            std::vector<float> values;
            float val;

            std::ifstream csv_file(voxel_path);
            std::string line;

            while (std::getline(csv_file, line)) {
                std::stringstream ss(line);
                while (ss >> val) {
                    values.emplace_back(val);
                    if (ss.peek() == ',') {
                        ss.ignore();
                    }
                }
                n_row += 1;
            }

            return torch::from_blob(values.data(), {1, 1, n_row, static_cast<unsigned int>(values.size()) / n_row}).clone();
        }
    };
} // namespace IRLMPNet

#endif
