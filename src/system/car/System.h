//
// Created by jiangeng on 12/31/20.
//

#ifndef IRLMPNET_SYSTEM_CAR_SYSTEM_H
#define IRLMPNET_SYSTEM_CAR_SYSTEM_H

#include <Eigen/Dense>
#include <ompl/control/SpaceInformation.h>

#include "system/car/StateSpace.h"
#include "system/car/ControlSpace.h"
#include "system/car/Propagator.h"
#include "system/car/StateValidityChecker.h"

namespace ob = ompl::base;
namespace oc = ompl::control;

namespace IRLMPNet {
    namespace System {
        class Car1OrderSystem {
        public:
            ob::StateSpacePtr state_space;
            oc::ControlSpacePtr control_space;
            oc::StatePropagatorPtr propagator;
            ob::StateValidityCheckerPtr collision_checker;
            oc::SpaceInformationPtr space_information;

            Car1OrderSystem() {
                state_space = std::make_shared<Car1OrderStateSpace>();
                control_space =  std::make_shared<Car1OrderControlSpace>(state_space);
                space_information = std::make_shared<oc::SpaceInformation>(state_space, control_space);
                propagator = std::make_shared<Car1OrderPropagator>(space_information);
                collision_checker = std::make_shared<CarCollisionChecker>(space_information, 0);
                space_information->setStatePropagator(propagator);
                space_information->setStateValidityChecker(collision_checker);
                space_information->setPropagationStepSize(0.1);
                space_information->setMinMaxControlDuration(1, 10); // TODO: check if this is good
            }
        };
    }
}

#endif //IRLMPNET_SYSTEM_CAR_SYSTEM_H
