//
// Created by jiangeng on 12/31/20.
//

#ifndef IRLMPNET_SYSTEM_CAR_CONTROLSPACE_H
#define IRLMPNET_SYSTEM_CAR_CONTROLSPACE_H

#include <ompl/control/spaces/RealVectorControlSpace.h>
#include "system/car/StateSpace.h"

namespace oc = ompl::control;
namespace ob = ompl::base;

namespace IRLMPNet {
    namespace System {
        class Car1OrderControlSpace : public oc::RealVectorControlSpace {
        public:
            using Ptr = std::shared_ptr<Car1OrderControlSpace>;

            explicit Car1OrderControlSpace(const ob::StateSpacePtr &space) : oc::RealVectorControlSpace(space, 2) {
                auto bounds = ob::RealVectorBounds(2);
                bounds.low = {-1, -1};
                bounds.high = {1, 1};
                setBounds(bounds);
            }
        };
    }
}

#endif //IRLMPNET_SYSTEM_CAR_CONTROLSPACE_H
