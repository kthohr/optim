.. Copyright (c) 2016-2022 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Differential Evolution
======================

**Table of contents**

.. contents:: :local:

----

Algorithm Description
---------------------

Differential Evolution (DE) is a stochastic genetic search algorithm for global optimization for problems of the form

.. math::

    \min_{x \in X} f(x)

where :math:`f` is potentially ill-behaved in one or more ways, such as non-convexity and/or non-differentiability.
The updating rule for DE is described below.

Let :math:`X^{(i)}` denote the :math:`N \times d` dimensional array of input values at stage :math:`i` of the algorithm, where each row corresponds to a different vector of candidate solutions.

1. The Mutation Step. For unique random indices :math:`a,b,c \in \{1, \ldots, d\}`, set the mutation proposal :math:`X^{(*)}` as follows.

   1. If ``de_mutation_method = 1``, use the 'rand' method:

      .. math::

        X^{(*)} = X^{(i)}(c,:) + F \times \left( X^{(i)}(a,:) - X^{(i)}(b,:) \right)

      where :math:`F` is the mutation parameter, set via ``de_par_F``.
    
   2. If ``de_mutation_method = 2``, use the 'best' method:

      .. math::

        X^{(*)} = X^{(i)}(\text{best},:) + F \times ( X^{(i)}(a,:) - X^{(i)}(b,:) )

      where

      .. math::

        X^{(i)} (\text{best},:) := \arg \min \left\{ f(X^{(i)}(1,:)), \ldots, f(X^{(i)}(N,:)) \right\}

2. The Crossover Step.

   1. Choose a random integer :math:`r_k \in \{1, \ldots, d \}`.

   2. Draw a vector :math:`u` of independent uniform random variables of length :math:`d`

   3. For each :math:`j \in \{ 1, \ldots, N \}` and :math:`k \in \{ 1, \ldots, d \}`, set

      .. math::

        X_c^{(*)} (j,k) = \begin{cases} X^*(j,k) & \text{ if } u_k \leq CR \text{ or } k = r_k \\ X^{(i)} (j,k) & \text{ else } \end{cases}

      where :math:`CR \in [0,1]` is the crossover parameter, set via ``de_par_CR``.

3. The Update Step.

      .. math::

        X^{(i+1)} (j,:) = \begin{cases} X_c^*(j,:) & \text{ if } f(X_c^*(j,:)) < f(X^{(i)}(j,:)) \\ X^{(i)} (j,:) & \text{ else } \end{cases}

The algorithm stops when at least one of the following conditions are met:

  1. the relative improvement in the objective function from the best candidate solution is less than ``rel_objfn_change_tol`` between ``de_settings.check_freq`` number of generations;
  
  2. the total number of generations exceeds ``de_settings.n_gen``.

----

DE with Population Reduction and Multiple Mutation Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TBW.

----

Function Declarations
---------------------

.. _de-func-ref1:
.. doxygenfunction:: de(ColVec_t&, std::function<fp_tconst ColVec_t &vals_inp, ColVec_t *grad_out, void *opt_data>, void *)
   :project: optimlib

.. _de-func-ref2:
.. doxygenfunction:: de(ColVec_t&, std::function<fp_tconst ColVec_t &vals_inp, ColVec_t *grad_out, void *opt_data>, void *, algo_settings_t&)
   :project: optimlib

----

.. _de-prmm-func-ref1:
.. doxygenfunction:: de_prmm(ColVec_t&, std::function<fp_tconst ColVec_t &vals_inp, ColVec_t *grad_out, void *opt_data>, void *)
   :project: optimlib

.. _de-prmm-func-ref2:
.. doxygenfunction:: de_prmm(ColVec_t&, std::function<fp_tconst ColVec_t &vals_inp, ColVec_t *grad_out, void *opt_data>, void *, algo_settings_t&)
   :project: optimlib


----

Optimization Control Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basic control parameters are:

- ``fp_t rel_objfn_change_tol``: the error tolerance value controlling how small the relative change in best candidate solution should be before 'convergence' is declared.

- ``size_t iter_max``: the maximum number of iterations/updates before the algorithm exits.

- ``bool vals_bound``: whether the search space of the algorithm is bounded. If ``true``, then

  - ``ColVec_t lower_bounds``: defines the lower bounds of the search space.

  - ``ColVec_t upper_bounds``: defines the upper bounds of the search space.

In addition to these:

- ``int print_level``: Set print level.

  - Level 1: Print iteration count and error value.

  - Level 2: Level 1 and print best input values and corresponding objective function value.

  - Level 3: Level 2 and print full population matrix, :math:`X`.

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
        
            bool success = optim::de(x, ackley_fn, nullptr);
        
            if (success) {
                std::cout << "de: Ackley test completed successfully." << std::endl;
            } else {
                std::cout << "de: Ackley test completed unsuccessfully." << std::endl;
            }
        
            arma::cout << "de: solution to Ackley test:\n" << x << arma::endl;
        
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
        
            bool success = optim::de(x, ackley_fn, nullptr);
        
            if (success) {
                std::cout << "de: Ackley test completed successfully." << std::endl;
            } else {
                std::cout << "de: Ackley test completed unsuccessfully." << std::endl;
            }
        
            arma::cout << "de: solution to Ackley test:\n" << x << arma::endl;
        
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
        
            bool success = optim::de(x, rastrigin_fn, &test_data);
        
            if (success) {
                std::cout << "de: Rastrigin test completed successfully." << std::endl;
            } else {
                std::cout << "de: Rastrigin test completed unsuccessfully." << std::endl;
            }
        
            arma::cout << "de: solution to Rastrigin test:\n" << x << arma::endl;
        
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
        
            bool success = optim::de(x, rastrigin_fn, &test_data);
        
            if (success) {
                std::cout << "de: Rastrigin test completed successfully." << std::endl;
            } else {
                std::cout << "de: Rastrigin test completed unsuccessfully." << std::endl;
            }
        
            arma::cout << "de: solution to Rastrigin test:\n" << x << arma::endl;
        
            return 0;
        }

----
