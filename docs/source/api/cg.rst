.. Copyright (c) 2016-2020 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Conjugate Gradient
==================

**Table of contents**

.. contents:: :local:

----

Algorithm Description
---------------------

The Nonlinear Conjugate Gradient algorithm is used solve to optimization problems of the form

.. math::

    \min_{x \in X} f(x)

where :math:`f` is convex and (at least) once differentiable. The algorithm requires the gradient function to be known. 

The updating rule for BFGS is described below. Let :math:`x^{(i)}` denote the function input values at stage :math:`i` of the algorithm.

1. Compute the descent direction using:

  .. math::

    d^{(i)} = - [\nabla_x f(x^{(i)})] + \beta^{(i)} d^{(i-1)}

2. Determine if :math:`\beta^{(i)}` should be reset (to zero), which occurs when

  .. math::

    \dfrac{| [\nabla f(x^{(i)})]^\top [\nabla f(x^{(i-1)})] |}{ [\nabla f(x^{(i)})]^\top [\nabla f(x^{(i)})] } > \nu
   
where :math:`\nu` is set via ``cg_settings.restart_threshold``.

3. Compute the optimal step size using line search:

  .. math::

    \alpha^{(i)} = \arg \min_{\alpha} f(x^{(i)} + \alpha \times d^{(i)})

4. Update the candidate solution vector using:

  .. math::

    x^{(i+1)} = x^{(i)} + \alpha^{(i)} \times d^{(i)}


The algorithm stops when one of the following conditions are ``true``:

1. the norm of the gradient vector, :math:`\| \nabla f \|`, is less than ``grad_err_tol``;

2. the relative change between :math:`x^{(i+1)}` and :math:`x^{(i)}` is less than ``rel_sol_change_tol``;

3. the total number of iterations exceeds ``iter_max``.

----

Updating Rules
~~~~~~~~~~~~~~

- ``cg_settings.method = 1`` Fletcher–Reeves (FR):

  .. math::

    \beta_{\text{FR}} = \dfrac{ [\nabla_x f(x^{(i)})]^\top [\nabla_x f(x^{(i)})] }{ [\nabla_x f(x^{(i-1)})]^\top [\nabla_x f(x^{(i-1)})] }

- ``cg_settings.method = 2`` Polak-Ribiere (PR):

  .. math::

    \beta_{\text{PR}} = \dfrac{ [\nabla_x f(x^{(i)})]^\top [\nabla_x f(x^{(i)})] }{ [\nabla_x f(x^{(i-1)})]^\top [\nabla_x f(x^{(i-1)})] }

- ``cg_settings.method = 3`` FR-PR Hybrid:

  .. math::

    \beta = \begin{cases} 
        - \beta_{\text{FR}} & \text{ if } \beta_{\text{PR}} < - \beta_{\text{FR}} \\ 
        \beta_{\text{PR}} & \text{ if } |\beta_{\text{PR}}| \leq \beta_{\text{FR}} \\
        \beta_{\text{FR}} & \text{ if } \beta_{\text{PR}} > \beta_{\text{FR}} \end{cases}

- ``cg_settings.method = 4`` Hestenes-Stiefel:

  .. math::

    \beta_{\text{HS}} = \dfrac{[\nabla_x f(x^{(i)})] \cdot ([\nabla_x f(x^{(i)})] - [\nabla_x f(x^{(i-1)})])}{([\nabla_x f(x^{(i)})] - [\nabla_x f(x^{(i-1)})]) \cdot d^{(i)}}

- ``cg_settings.method = 5`` Dai-Yuan:

  .. math::

    \beta_{\text{DY}} = \dfrac{[\nabla_x f(x^{(i)})] \cdot [\nabla_x f(x^{(i)})]}{([\nabla_x f(x^{(i)})] - [\nabla_x f(x^{(i-1)})]) \cdot d^{(i)}}

- ``cg_settings.method = 6`` Hager-Zhang:

  .. math::

    \begin{aligned}
    \beta_{\text{HZ}} &= \left( y^{(i)} - 2 \times \dfrac{[y^{(i)}] \cdot y^{(i)}}{y^{(i)} \cdot d^{(i)}} \times d^{(i)} \right) \cdot \dfrac{[\nabla_x f(x^{(i)})]}{y^{(i)} \cdot d^{(i)}} \\ 
    y^{(i)} &:= [\nabla_x f(x^{(i)})] - [\nabla_x f(x^{(i-1)})]
    \end{aligned}

Finally, we set: 

.. math::
  \beta^{(i)} = \max \{ 0, \beta_{*} \}


where :math:`\beta_{*}` is the update method chosen.

----

Function Declarations
---------------------

.. _cg-func-ref1:
.. doxygenfunction:: cg(Vec_t&, std::function<doubleconst Vec_t &vals_inp, Vec_t *grad_out, void *opt_data>, void *)
   :project: optimlib

.. _cg-func-ref2:
.. doxygenfunction:: cg(Vec_t&, std::function<doubleconst Vec_t &vals_inp, Vec_t *grad_out, void *opt_data>, void *, algo_settings_t&)
   :project: optimlib

----

Optimization Control Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basic control parameters are:

- ``double grad_err_tol``: the error tolerance value controlling how small the :math:`L_2` norm of the gradient vector :math:`\| \nabla f \|` should be before 'convergence' is declared.

- ``double rel_sol_change_tol``: the error tolerance value controlling how small the proportional change in the solution vector should be before 'convergence' is declared.

  The relative change is computed using:

    .. math::

        \left\| \dfrac{x^{(i)} - x^{(i-1)}}{ |x^{(i-1)}| + \epsilon } \right\|_1

  where :math:`\epsilon` is a small number added for numerical stability.

- ``size_t iter_max``: the maximum number of iterations/updates before the algorithm exits.

- ``bool vals_bound``: whether the search space of the algorithm is bounded. If ``true``, then

  - ``Vec_t lower_bounds``: defines the lower bounds of the search space.

  - ``Vec_t upper_bounds``: defines the upper bounds of the search space.

Additional settings:

- ``int cg_settings.method``: Update method.

  - Default value: ``2``.

- ``double cg_settings.restart_threshold``: parameter :math:`\nu` from step 2 in the algorithm description.

  - Default value: ``0.1``.

- ``bool use_rel_sol_change_crit``: whether to enable the ``rel_sol_change_tol`` stopping criterion.

  - Default value: ``false``.

- ``double cg_settings.wolfe_cons_1``: Line search tuning parameter that controls the tolerance on the Armijo sufficient decrease condition.

  - Default value: ``1E-03``.

- ``double cg_settings.wolfe_cons_2``: Line search tuning parameter that controls the tolerance on the curvature condition.

  - Default value: ``0.10``.

- ``int print_level``: Set the level of detail for printing updates on optimization progress.

  - Level ``0``: Nothing (default).

  - Level ``1``: Print the iteration count and current error values.

  - Level ``2``: Level 1 plus the current candidate solution values, :math:`x^{(i+1)}`.

  - Level ``3``: Level 2 plus the direction vector, :math:`d^{(i)}`, and the gradient vector, :math:`\nabla_x f(x^{(i+1)})`.

  - Level ``4``: Level 3 plus :math:`\beta^{(i)}`.

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
        
            bool success = optim::cg(x, sphere_fn, nullptr);
        
            if (success) {
                std::cout << "cg: sphere test completed successfully." << "\n";
            } else {
                std::cout << "cg: sphere test completed unsuccessfully." << "\n";
            }
        
            arma::cout << "cg: solution to sphere test:\n" << x << arma::endl;
        
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
        
            bool success = optim::cg(x, sphere_fn, nullptr);
        
            if (success) {
                std::cout << "cg: sphere test completed successfully." << "\n";
            } else {
                std::cout << "cg: sphere test completed unsuccessfully." << "\n";
            }
        
            std::cout << "cg: solution to sphere test:\n" << x << std::endl;
        
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
        
            bool success_2 = optim::cg(x, booth_fn, nullptr);
        
            if (success_2) {
                std::cout << "cg: Booth test completed successfully." << "\n";
            } else {
                std::cout << "cg: Booth test completed unsuccessfully." << "\n";
            }
        
            arma::cout << "cg: solution to Booth test:\n" << x_2 << arma::endl;
        
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
        
            bool success_2 = optim::cg(x, booth_fn, nullptr);
        
            if (success_2) {
                std::cout << "cg: Booth test completed successfully." << "\n";
            } else {
                std::cout << "cg: Booth test completed unsuccessfully." << "\n";
            }
        
            std::cout << "cg: solution to Booth test:\n" << x_2 << std::endl;
        
            return 0;
        }

----
