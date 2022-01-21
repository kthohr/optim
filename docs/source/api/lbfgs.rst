.. Copyright (c) 2016-2022 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Limited Memory BFGS
===================

**Table of contents**

.. contents:: :local:

----

Algorithm Description
---------------------

The limited memory variant of the Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) algorithm is a quasi-Newton optimization method that can be used solve to optimization problems of the form

.. math::

    \min_{x \in X} f(x)

where :math:`f` is convex and twice differentiable. The BFGS algorithm requires that the gradient of :math:`f` be known and forms an approximation to the Hessian. 

The updating rule for L-BFGS is described below. Let :math:`x^{(i)}` denote the candidate solution vector at stage :math:`i` of the algorithm.

1. Compute the descent direction using a two loop recursion. Let

   .. math::

        \begin{aligned}
            s^{(i)} &:= x^{(i+1)} - x^{(i)} \\
            y^{(i)} &:= \nabla_x f(x^{(i+1)}) - \nabla_x f(x^{(i)})
        \end{aligned}

   Then

   a. Set

      .. math::

        q = \nabla_x f(x^{(i)})

   b. For :math:`k = i - 1, \ldots, i - M` do:

      .. math::
        \begin{aligned}
            \rho_k &:= \frac{1}{ [y^{(k)}]^\top s^{(k)} } \\
            \gamma_k &:= \rho_k \times [s^{(k)}]^\top q \\
            q &= q - \gamma_k \times y^{(k)}
        \end{aligned}

      where :math:`M`, the history length, is set via ``par_M``.

   c. Set

      .. math::

        r = q \times \frac{[s^{(i)}]^\top y^{(i)}}{[y^{(i)}]^\top y^{(i)}}

   d. For :math:`k = i - M, \ldots, i - 1` do:

      .. math::

        \begin{aligned}
            \beta &:= \rho_k \times [y^{(k)}]^\top r \\
            r &= r + s^{(k)} (\gamma_k - \beta)
        \end{aligned}
    
   e. Set

      .. math::

        d^{(i)} = - r

2. Compute the optimal step size using line search:

   .. math::

        \alpha^{(i)} = \arg \min_{\alpha} f(x^{(i)} + \alpha \times d^{(i)})

3. Update the candidate solution vector using:

   .. math::

        x^{(i+1)} = x^{(i)} + \alpha^{(i)} \times d^{(i)}


The algorithm stops when at least one of the following conditions are met:

  1. the norm of the gradient vector, :math:`\| \nabla f \|`, is less than ``grad_err_tol``;

  2. the relative change between :math:`x^{(i+1)}` and :math:`x^{(i)}` is less than ``rel_sol_change_tol``;

  3. the total number of iterations exceeds ``iter_max``.

----

Function Declarations
---------------------

.. _lbfgs-func-ref1:
.. doxygenfunction:: lbfgs(ColVec_t&, std::function<fp_tconst ColVec_t &vals_inp, ColVec_t *grad_out, void *opt_data>, void *)
   :project: optimlib

.. _lbfgs-func-ref2:
.. doxygenfunction:: lbfgs(ColVec_t&, std::function<fp_tconst ColVec_t &vals_inp, ColVec_t *grad_out, void *opt_data>, void *, algo_settings_t&)
   :project: optimlib

----

Optimization Control Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basic control parameters are:

- ``fp_t grad_err_tol``: the error tolerance value controlling how small the :math:`L_2` norm of the gradient vector :math:`\| \nabla f \|` should be before 'convergence' is declared.

- ``fp_t rel_sol_change_tol``: the error tolerance value controlling how small the proportional change in the solution vector should be before 'convergence' is declared.

  The relative change is computed using:

    .. math::

        \left\| \dfrac{x^{(i)} - x^{(i-1)}}{ |x^{(i-1)}| + \epsilon } \right\|_1

  where :math:`\epsilon` is a small number added for numerical stability.

- ``size_t iter_max``: the maximum number of iterations/updates before the algorithm exits.

- ``bool vals_bound``: whether the search space of the algorithm is bounded. If ``true``, then

  - ``ColVec_t lower_bounds``: defines the lower bounds of the search space.

  - ``ColVec_t upper_bounds``: defines the upper bounds of the search space.

Additional settings:

- ``size_t lbfgs_settings.par_M``: The number of past gradient vectors to use when forming the approximate Hessian matrix.

  - Default value: ``10``.

- ``fp_t lbfgs_settings.wolfe_cons_1``: Line search tuning parameter that controls the tolerance on the Armijo sufficient decrease condition.

  - Default value: ``1E-03``.

- ``fp_t lbfgs_settings.wolfe_cons_2``: Line search tuning parameter that controls the tolerance on the curvature condition.

  - Default value: ``0.90``.

- ``int print_level``: Set the level of detail for printing updates on optimization progress.

  - Level ``0``: Nothing (default).

  - Level ``1``: Print the current iteration count and error values.

  - Level ``2``: Level 1 plus the current candidate solution values, :math:`x^{(i+1)}`.

  - Level ``3``: Level 2 plus the direction vector, :math:`d^{(i)}`, and the gradient vector, :math:`\nabla_x f(x^{(i+1)})`.

  - Level ``4``: Level 3 plus print components used to update the approximate inverse Hessian matrix: :math:`s` and :math:`y`.

----

Examples
--------

Sphere Function
~~~~~~~~~~~~~~~

Code to run this example is given below.

.. toggle-header::
    :header: **Armadillo (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_ARMA_WRAPPERS
        #include "optim.hpp"
        
        inline
        double 
        sphere_fn(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
        {
            double obj_val = arma::dot(vals_inp,vals_inp);
            
            if (grad_out) {
                *grad_out = 2.0*vals_inp;
            }
            
            return obj_val;
        }
        
        int main()
        {
            const int test_dim = 5;
        
            arma::vec x = arma::ones(test_dim,1); // initial values (1,1,...,1)
        
            bool success = optim::lbfgs(x, sphere_fn, nullptr);
        
            if (success) {
                std::cout << "bfgs: sphere test completed successfully." << "\n";
            } else {
                std::cout << "bfgs: sphere test completed unsuccessfully." << "\n";
            }
        
            arma::cout << "bfgs: solution to sphere test:\n" << x << arma::endl;
        
            return 0;
        }

.. toggle-header::
    :header: **Eigen (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_EIGEN_WRAPPERS
        #include "optim.hpp"
        
        inline
        double 
        sphere_fn(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* opt_data)
        {
            double obj_val = vals_inp.dot(vals_inp);
            
            if (grad_out) {
                *grad_out = 2.0*vals_inp;
            }
            
            return obj_val;
        }
        
        int main()
        {
            const int test_dim = 5;
        
            Eigen::VectorXd x = Eigen::VectorXd::Ones(test_dim); // initial values (1,1,...,1)
        
            bool success = optim::lbfgs(x, sphere_fn, nullptr);
        
            if (success) {
                std::cout << "bfgs: sphere test completed successfully." << "\n";
            } else {
                std::cout << "bfgs: sphere test completed unsuccessfully." << "\n";
            }
        
            std::cout << "bfgs: solution to sphere test:\n" << x << std::endl;
        
            return 0;
        }

----

Booth's Function
~~~~~~~~~~~~~~~~

Code to run this example is given below.

.. toggle-header::
    :header: **Armadillo Code (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_ARMA_WRAPPERS
        #include "optim.hpp"

        inline
        double 
        booth_fn(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
        {
            double x_1 = vals_inp(0);
            double x_2 = vals_inp(1);
        
            double obj_val = std::pow(x_1 + 2*x_2 - 7.0,2) + std::pow(2*x_1 + x_2 - 5.0,2);
            
            if (grad_out) {
                (*grad_out)(0) = 10*x_1 + 8*x_2   2*(- 7.0) + 4*(x_2 - 5.0);
                (*grad_out)(1) = 2*(x_1 + 2*x_2 - 7.0)*2 + 2*(2*x_1 + x_2 - 5.0);
            }
            
            return obj_val;
        }
        
        int main()
        {        
            arma::vec x_2 = arma::zeros(2,1); // initial values (0,0)
        
            bool success_2 = optim::lbfgs(x, booth_fn, nullptr);
        
            if (success_2) {
                std::cout << "bfgs: Booth test completed successfully." << "\n";
            } else {
                std::cout << "bfgs: Booth test completed unsuccessfully." << "\n";
            }
        
            arma::cout << "bfgs: solution to Booth test:\n" << x_2 << arma::endl;
        
            return 0;
        }

.. toggle-header::
    :header: **Eigen Code (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_EIGEN_WRAPPERS
        #include "optim.hpp"

        inline
        double 
        booth_fn(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* opt_data)
        {
            double x_1 = vals_inp(0);
            double x_2 = vals_inp(1);
        
            double obj_val = std::pow(x_1 + 2*x_2 - 7.0,2) + std::pow(2*x_1 + x_2 - 5.0,2);
            
            if (grad_out) {
                (*grad_out)(0) = 2*(x_1 + 2*x_2 - 7.0) + 2*(2*x_1 + x_2 - 5.0)*2;
                (*grad_out)(1) = 2*(x_1 + 2*x_2 - 7.0)*2 + 2*(2*x_1 + x_2 - 5.0);
            }
            
            return obj_val;
        }
        
        int main()
        {        
            Eigen::VectorXd x = Eigen::VectorXd::Zero(test_dim); // initial values (0,0)
        
            bool success_2 = optim::lbfgs(x, booth_fn, nullptr);
        
            if (success_2) {
                std::cout << "bfgs: Booth test completed successfully." << "\n";
            } else {
                std::cout << "bfgs: Booth test completed unsuccessfully." << "\n";
            }
        
            std::cout << "bfgs: solution to Booth test:\n" << x_2 << std::endl;
        
            return 0;
        }

----
