#ifndef IRLMPNET_PLANNER_CONVERTER_H_
#define IRLMPNET_PLANNER_CONVERTER_H_

#include <torch/script.h>

namespace IRLMPNet {
    /// convert from 2d single-precision tensor to vector. 
    /// The tensor is 2d, because the first dim is batch size. 
    inline std::vector<double> toVector(const torch::Tensor &tensor, const unsigned int dim) {
        auto data = tensor.accessor<float, 2>()[0];
        std::vector<double> dest(dim); // TODO: maybe we can optimize it by using a fixed size array
        for (unsigned int i = 0; i < dim; i++) {
            dest[i] = static_cast<float>(data[i]);
        }
        return dest;
    }

    /// convert from vector to 2d single-precision tensor.
    /// The tensor is 2d, because the first dim is batch size. 
    inline torch::Tensor toTensor(const std::vector<double> &vec, const unsigned int dim) {
        std::vector<float> scaled_src(dim); // TODO: maybe we can optimize it by using a fixed size array
        for (unsigned int i = 0; i < dim; i++) {
            scaled_src[i] = vec[i];
        }
        return torch::from_blob(scaled_src.data(), {1, dim}).clone();
    }
} // namespace IRLMPNet


#endif