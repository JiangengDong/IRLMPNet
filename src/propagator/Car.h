#ifndef IRLMPNET_PROPAGATOR_CAR_H_
#define IRLMPNET_PROPAGATOR_CAR_H_

#include <Eigen/Dense>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/ODESolver.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/StatePropagator.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <propagator/ODESolver.h>

namespace oc = ompl::control;
namespace ob = ompl::base;

namespace IRLMPNet {

/// \brief we assume the state space be 3-dim RealVectorSpace, and the control space be 2-dim RealVectorSpace
class DifferentialCar1OrderPropagator : public oc::StatePropagator {
public:
    ob::StateSpacePtr space_;
    double time_step_;

    DifferentialCar1OrderPropagator(const oc::SpaceInformationPtr &si) : oc::StatePropagator(si) {
        space_ = si_->getStateSpace();
        time_step_ = 0.01;
    }

    void propagate(const ob::State *state, const oc::Control *control, double duration, ob::State *result) const override {
        space_->copyState(result, state);

        auto pstate = state->as<ob::RealVectorStateSpace::StateType>()->values;
        auto pcontrol = control->as<oc::RealVectorControlSpace::ControlType>()->values;
        auto presult = result->as<ob::RealVectorStateSpace::StateType>()->values;

        RK4(pstate, 3, pcontrol, duration, &ode, presult);
        presult[2] = remainder(presult[2], 2 * M_PI);
        space_->enforceBounds(result);
    }

private:
    /// system dynamics
    static void ode(const double *state, const double *control, double *result) {
        const double &x = state[0], &y = state[1], &theta = state[2];
        const double &v = control[0], &w = control[1];

        result[0] = v * cos(theta);
        result[1] = v * sin(theta);
        result[2] = w;
    }
};

/// \brief we assume the state space be 5-dim RealVectorSpace, and the control space be 2-dim RealVectorSpace
class DifferentialCar2OrderPropagator : public oc::StatePropagator {
public:
    ob::StateSpacePtr space_;
    double time_step_;

    DifferentialCar2OrderPropagator(const oc::SpaceInformationPtr &si) : oc::StatePropagator(si) {
        space_ = si_->getStateSpace();
        time_step_ = 0.01;
    }

    void propagate(const ob::State *state, const oc::Control *control, double duration, ob::State *result) const override {
        space_->copyState(result, state);

        auto pstate = state->as<ob::RealVectorStateSpace::StateType>()->values;
        auto pcontrol = control->as<oc::RealVectorControlSpace::ControlType>()->values;
        auto presult = result->as<ob::RealVectorStateSpace::StateType>()->values;

        RK4(pstate, 5, pcontrol, duration, &ode, presult);
        presult[2] = remainder(presult[2], 2 * M_PI);
        space_->enforceBounds(result);
    }

private:
    /// system dynamics
    static void ode(const double *state, const double *control, double *result) {
        const double &x = state[0], &y = state[1], &theta = state[2], &v = state[3], &w = state[4];
        const double &lin_accel = control[0], &ang_accel = control[1];

        result[0] = v * cos(theta);
        result[1] = v * sin(theta);
        result[2] = w;
        result[3] = lin_accel;
        result[4] = ang_accel;
    }
};

} // namespace IRLMPNet

#endif