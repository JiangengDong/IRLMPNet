#ifndef IRLMPNET_ODESOLVER_H_
#define IRLMPNET_ODESOLVER_H_

#include <iostream>

namespace IRLMPNet {
    namespace System {
        /// add two vector together
        inline void addVec(const double *a, const double *b, const size_t dim, double *c) {
            for (size_t i = 0; i < dim; i++) {
                c[i] = a[i] + b[i];
            }
        }

        /// multiply a vector with a scalar
        inline void mulScalar(const double *a, const double b, const size_t dim, double *c) {
            for (size_t i = 0; i < dim; i++) {
                c[i] = a[i] * b;
            }
        }

        /// copy a vector from source to destination
        inline void copyVec(const double *a, const size_t dim, double *b) {
            for (size_t i = 0; i < dim; i++) {
                b[i] = a[i];
            }
        }

        using ODEFunction = void (*)(const double *, const double *, double *);

        inline void RK4(const double *state, const size_t state_dim, const double *control, const double dt, const ODEFunction ode, double *result) {
            // WARNING: I think 16 is enough for most of the state space.
            //          If not, increase it manually. I prefer to fix the size at compile time to save time.
            double k1[16], k2[16], k3[16], k4[16], temp[16];

            // k1 = ode(state, control)*dt
            (*ode)(state, control, k1);
            mulScalar(k1, dt, state_dim, k1);

            // k2 = ode(state + k1 * 0.5, control)*dt
            mulScalar(k1, 0.5, state_dim, temp);
            addVec(state, temp, state_dim, temp);
            (*ode)(temp, control, k2);
            mulScalar(k2, dt, state_dim, k2);

            // k3 = ode(state + k2 * 0.5, control)*dt
            mulScalar(k2, 0.5, state_dim, temp);
            addVec(state, temp, state_dim, temp);
            (*ode)(temp, control, k3);
            mulScalar(k3, dt, state_dim, k3);

            // k4 = ode(state + k3 * 0.5, control)*dt
            mulScalar(k3, 0.5, state_dim, temp);
            addVec(state, temp, state_dim, temp);
            (*ode)(temp, control, k4);
            mulScalar(k4, dt, state_dim, k4);

            // result = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            mulScalar(k1, 1.0 / 6.0, state_dim, k1);
            mulScalar(k2, 1.0 / 3.0, state_dim, k2);
            mulScalar(k3, 1.0 / 3.0, state_dim, k3);
            mulScalar(k4, 1.0 / 6.0, state_dim, k4);

            copyVec(state, state_dim, result);
            addVec(result, k1, state_dim, result);
            addVec(result, k2, state_dim, result);
            addVec(result, k3, state_dim, result);
            addVec(result, k4, state_dim, result);
        }
    }
} // namespace IRLMPNet

#endif