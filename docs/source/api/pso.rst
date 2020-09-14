.. Copyright (c) 2016-2020 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Particle Swarm Optimization
===========================

**Table of contents**

.. contents:: :local:

----

Algorithm Description
---------------------

Particle Swarm Optimization (PSO) is a stochastic swarm intelligence algorithm for global optimization

.. math::

    \min_{x \in X} f(x)

where :math:`f` is potentially ill-behaved in one or more ways, such as non-convexity and/or non-differentiability.
The updating rule for PSO is described below.

Let :math:`X^{(i)}` denote the :math:`N \times d` dimensional array of input values at stage :math:`i` of the algorithm, where each row corresponds to a different vector of candidate solutions.

1. Update the velocity and position matrices. Sample two :math:`d`-dimensional iid uniform random vectors, :math:`R_C, R_S`.

   Update each velocity vector using:

   .. math::

      V^{(i+1)}(j.:) = w V^{(i+1)}(j,:) + c_C \times R_C \odot (X_b^{(i)} (j,:) - X^{(i)}(j,:)) + c_S \times R_S \odot (g_b - X^{(i)}(j,:))
    
   Each position vector is updated using:

   .. math::

      X^{(i+1)}(j,:) = X^{(i)}(j,:) + V^{(i+1)}(j,:)

2. Update local-best particle.

   .. math::

      X_b^{(i+1)}(j,:) = \begin{cases} X^{(i+1)}(j,:) & \text{ if } f(X^{(i+1)}(j,:)) < f(X_b^{(i)}(j,:)) \\ X_b^{(i)}(j,:) & \text{ else } \end{cases}

3. Update the global-best particle.

   Let

   .. math::

      j^{(*)} = \arg \min_{j \in \{1, \ldots, N\}} f(X^{(i+1)} (j,:))

   Then

   .. math::

      g_b = \begin{cases} X^{(i+1)}(j^{(*)},:) & \text{ if } f(X^{(i+1)}(j^{(*)},:)) < f(g_b) \\ g_b & \text{ else } \end{cases}


The algorithm stops when one of the following conditions are ``true``:

  1. the relative improvement in the objective function is less than ``rel_objfn_change_tol`` between ``pso_settings.check_freq`` number of generations;
  2. the total number of generations exceeds ``pso_settings.n_gen``.

----

PSO with Differentially-Perturbed Velocity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TBW.

----

Function Declarations
---------------------

.. _pso-func-ref1:
.. doxygenfunction:: pso(Vec_t&, std::function<doubleconst Vec_t &vals_inp, Vec_t *grad_out, void *opt_data>, void *)
   :project: optimlib

.. _pso-func-ref2:
.. doxygenfunction:: pso(Vec_t&, std::function<doubleconst Vec_t &vals_inp, Vec_t *grad_out, void *opt_data>, void *, algo_settings_t&)
   :project: optimlib

----

.. _pso-dv-func-ref1:
.. doxygenfunction:: pso_dv(Vec_t&, std::function<doubleconst Vec_t &vals_inp, Vec_t *grad_out, void *opt_data>, void *)
   :project: optimlib

.. _pso-dv-func-ref2:
.. doxygenfunction:: pso_dv(Vec_t&, std::function<doubleconst Vec_t &vals_inp, Vec_t *grad_out, void *opt_data>, void *, algo_settings_t&)
   :project: optimlib

----

Optimization Control Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basic control parameters are:

- ``double rel_objfn_change_tol``: the error tolerance value controlling how small the relative change in best candidate solution should be before 'convergence' is declared.

- ``size_t iter_max``: the maximum number of iterations/updates before the algorithm exits.

- ``bool vals_bound``: whether the search space of the algorithm is bounded. If ``true``, then

  - ``Vec_t lower_bounds``: defines the lower bounds of the search space.

  - ``Vec_t upper_bounds``: defines the upper bounds of the search space.

In addition to these:

- ``int print_level``: Set print level.

  - Level 1: Print iteration count and error value.

  - Level 2: Level 1 and print best input values, as well as objective function values.

  - Level 3: Level 2 and print full matrix :math:`X`.

----

Examples
--------

Ackley Function
~~~~~~~~~~~~~~~

Code to run this example is given below.

.. toggle-header::
    :header: **Armadillo (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_ARMA_WRAPPERS
        #include "optim.hpp"
        
        #define OPTIM_PI 3.14159265358979

        double 
        ackley_fn(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
        {
            const double x = vals_inp(0);
            const double y = vals_inp(1);

            double obj_val = 20 + std::exp(1) - 20*std::exp( -0.2*std::sqrt(0.5*(x*x + y*y)) ) - std::exp( 0.5*(std::cos(2 * OPTIM_PI * x) + std::cos(2 * OPTIM_PI * y)) );
            
            return obj_val;
        }
        
        int main()
        {
            arma::vec x = arma::ones(2,1) + 1.0; // initial values: (2,2)
        
            bool success = optim::pso(x, ackley_fn, nullptr);
        
            if (success) {
                std::cout << "pso: Ackley test completed successfully." << std::endl;
            } else {
                std::cout << "pso: Ackley test completed unsuccessfully." << std::endl;
            }
        
            arma::cout << "pso: solution to Ackley test:\n" << x << arma::endl;
        
            return 0;
        }

.. toggle-header::
    :header: **Eigen (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_EIGEN_WRAPPERS
        #include "optim.hpp"
        
        #define OPTIM_PI 3.14159265358979

        double 
        ackley_fn(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* opt_data)
        {
            const double x = vals_inp(0);
            const double y = vals_inp(1);

            double obj_val = 20 + std::exp(1) - 20*std::exp( -0.2*std::sqrt(0.5*(x*x + y*y)) ) - std::exp( 0.5*(std::cos(2 * OPTIM_PI * x) + std::cos(2 * OPTIM_PI * y)) );
            
            return obj_val;
        }
        
        int main()
        {
            Eigen::VectorXd x = 2.0 * Eigen::VectorXd::Ones(2); // initial values: (2,2)
        
            bool success = optim::pso(x, ackley_fn, nullptr);
        
            if (success) {
                std::cout << "pso: Ackley test completed successfully." << std::endl;
            } else {
                std::cout << "pso: Ackley test completed unsuccessfully." << std::endl;
            }
        
            arma::cout << "pso: solution to Ackley test:\n" << x << arma::endl;
        
            return 0;
        }

----

Rastrigin Function
~~~~~~~~~~~~~~~~~~

Code to run this example is given below.

.. toggle-header::
    :header: **Armadillo Code (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_ARMA_WRAPPERS
        #include "optim.hpp"

        #define OPTIM_PI 3.14159265358979
 
        struct rastrigin_fn_data {
            double A;
        };
        
        double 
        rastrigin_fn(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
        {
            const int n = vals_inp.n_elem;
        
            rastrigin_fn_data* objfn_data = reinterpret_cast<rastrigin_fn_data*>(opt_data);
            const double A = objfn_data->A;
        
            double obj_val = A*n + arma::accu( arma::pow(vals_inp,2) - A*arma::cos(2 * OPTIM_PI * vals_inp) );
            
            return obj_val;
        }
        
        int main()
        {
            rastrigin_fn_data test_data;
            test_data.A = 10;
        
            arma::vec x = arma::ones(2,1) + 1.0; // initial values: (2,2)
        
            bool success = optim::pso(x, rastrigin_fn, &test_data);
        
            if (success) {
                std::cout << "pso: Rastrigin test completed successfully." << std::endl;
            } else {
                std::cout << "pso: Rastrigin test completed unsuccessfully." << std::endl;
            }
        
            arma::cout << "pso: solution to Rastrigin test:\n" << x << arma::endl;
        
            return 0;
        }

.. toggle-header::
    :header: **Eigen Code (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_EIGEN_WRAPPERS
        #include "optim.hpp"

        #define OPTIM_PI 3.14159265358979
 
        struct rastrigin_fn_data {
            double A;
        };
        
        double 
        rastrigin_fn(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* opt_data)
        {
            const int n = vals_inp.n_elem;
        
            rastrigin_fn_data* objfn_data = reinterpret_cast<rastrigin_fn_data*>(opt_data);
            const double A = objfn_data->A;
        
            double obj_val = A*n + vals_inp.array().pow(2).sum() - A * (2 * OPTIM_PI * vals_inp).array().cos().sum();
            
            return obj_val;
        }
        
        int main()
        {
            rastrigin_fn_data test_data;
            test_data.A = 10;
        
            Eigen::VectorXd x = 2.0 * Eigen::VectorXd::Ones(2); // initial values: (2,2)
        
            bool success = optim::pso(x, rastrigin_fn, &test_data);
        
            if (success) {
                std::cout << "pso: Rastrigin test completed successfully." << std::endl;
            } else {
                std::cout << "pso: Rastrigin test completed unsuccessfully." << std::endl;
            }
        
            arma::cout << "pso: solution to Rastrigin test:\n" << x << arma::endl;
        
            return 0;
        }

----
