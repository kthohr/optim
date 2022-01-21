.. Copyright (c) 2016-2022 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Nelder-Mead
===========

**Table of contents**

.. contents:: :local:

----

Algorithm Description
---------------------

Nelder-Mead is a derivative-free simplex method, used solve to optimization problems of the form

.. math::

    \min_{x \in X} f(x)

where :math:`f` need not be convex or differentiable. 

The updating rule for Nelder-Mead is described below. Let :math:`x^{(i)}` denote the simplex values at stage :math:`i` of the algorithm.

1. Sort the simplex vertices :math:`x` in order of function values, from smallest to largest:

   .. math::

        f(x^{(i)})(1,:) \leq f(x^{(i)})(2,:) \leq \cdots \leq f(x^{(i)})(n+1,:)

2. Calculate the centroid value up to the :math:`n` th vertex:

   .. math::

        \bar{x} = \frac{1}{n} \sum_{j=1}^n x^{(i)}(j,:)

   and compute the reflection point:

   .. math::

        x^r = \bar{x} + \alpha (\bar{x} - x^{(i)}(n+1,:))

   where :math:`\alpha` is set by ``par_alpha``.

   If :math:`f(x^{(i)}(1,:)) \leq f(x^r) < f(x^{(i)}(n,:))`, then
   
   .. math::
        
        x^{(i+1)}(n+1,:) = x^r, \ \ \textbf{ and go to Step 1.}

   Otherwise continue to Step 3.

3. If :math:`f(x^r) \geq f(x^{(i)}(1,:))` then go to Step 4, otherwise compute the expansion point:

   .. math::

        x^e = \bar{x} + \gamma (x^r - \bar{x})
    
   where :math:`\gamma` is set by ``par_gamma``.

   Set

   .. math::

        x^{(i+1)}(n+1,:) = \begin{cases} x^e & \text{ if } f(x^e) < f(x^r) \\ x^r & \text{ else } \end{cases}

   and go to Step 1.

4. If :math:`f(x^r) < f(x^{(i)}(n,:))` then compute the outside or inside contraction:

   .. math::

        x^{c} = \begin{cases} \bar{x} + \beta(x^r - \bar{x}) & \text{ if } f(x^r) < f(x^{(i)}(n+1,:)) \\ \bar{x} - \beta(x^r - \bar{x}) & \text{ else} \end{cases}

   If :math:`f(x^c) < f(x^{(i)}(n+1,:))`, then

   .. math::

        x^{(i+1)}(n+1,:) = x^c, \ \ \textbf{ and go to Step 1.}

   Otherwise go to Step 5.

5. Shrink the simplex toward :math:`x^{(i)}(1,:)`:

   .. math::

        x^{(i+1)}(j,:) = x^{(i)}(1,:) + \delta (x^{(i)}(j,:) - x^{(i)}(1,:)), \ \ j = 2, \ldots, n+1

   where :math:`\delta` is set by ``par_delta``. Go to Step 1.


The algorithm stops when at least one of the following conditions are met:

  1. the relative change in the simplex of function values, defined as:

     .. math::

        \dfrac{\max \{ | f(x^{(i+1)}(1,:)) - f(x^{(i)}(1,:)) |, | f(x^{(i+1)}(n+1,:)) - f(x^{(i)}(1,:)) | \} }{ \max_j | f(x^{(i+1)}(j,:)) | + \epsilon};

     is less than ``rel_objfn_change_tol``.

  2. the relative change between :math:`x^{(i+1)}` and :math:`x^{(i)}` is less than ``rel_sol_change_tol``;

  3. the total number of iterations exceeds ``iter_max``.

----

Function Declarations
---------------------

.. _nm-func-ref1:
.. doxygenfunction:: nm(ColVec_t&, std::function<fp_tconst ColVec_t &vals_inp, ColVec_t *grad_out, void *opt_data>, void *)
   :project: optimlib

.. _nm-func-ref2:
.. doxygenfunction:: nm(ColVec_t&, std::function<fp_tconst ColVec_t &vals_inp, ColVec_t *grad_out, void *opt_data>, void *, algo_settings_t&)
   :project: optimlib

----

Optimization Control Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basic control parameters are:

- ``fp_t rel_objfn_change_tol``: the error tolerance value controlling how small the relative change in the simplex of function values, defined as:

    .. math::

        \dfrac{\max \{ | f(x^{(i+1)}(1,:)) - f(x^{(i)}(1,:)) |, | f(x^{(i+1)}(n+1,:)) - f(x^{(i)}(1,:)) | \} }{ \max_j | f(x^{(i+1)}(j,:)) | + \epsilon};
 

  should be before 'convergence' is declared.

- ``fp_t rel_sol_change_tol``: the error tolerance value controlling how small the proportional change in the solution vector should be before 'convergence' is declared.

  The relative change is computed using:

    .. math::

       \dfrac{\max_{j,k}|x^{(i+1)}(j,k) - x^{(i)}(j,k)|}{ \max_{j,k}|x^{(i)}(j,k)| + \epsilon }

  where :math:`\epsilon` is a small number added for numerical stability.

- ``size_t iter_max``: the maximum number of iterations/updates before the algorithm exits.

- ``bool vals_bound``: whether the search space of the algorithm is bounded. If ``true``, then

  - ``ColVec_t lower_bounds``: defines the lower bounds of the search space.

  - ``ColVec_t upper_bounds``: defines the upper bounds of the search space.

- ``struct nm_settings_t``, which defines several parameters that control the behavior of the simplex.

  - ``bool adaptive_pars = true``: scale the contraction, expansion, and shrinkage parameters using the dimension of the optimization problem.

  - ``fp_t par_alpha = 1.0``: reflection parameter.

  - ``fp_t par_beta = 0.5``: contraction parameter.

  - ``fp_t par_gamma = 2.0``: expansion parameter.

  - ``fp_t par_delta = 0.5``: shrinkage parameter.

  - ``bool custom_initial_simplex = false``: whether to use user-defined values for the initial simplex matrix.

  - ``Mat_t initial_simplex_points``: user-defined values for the initial simplex (optional). Dimensions: :math:`(n + 1) \times n`.

In addition to these:

- ``int print_level``: Set the level of detail for printing updates on optimization progress.

  - Level ``1``: Print the iteration count and current error values.

  - Level ``2``: Level 1 plus the current candidate solution values.

  - Level ``3``: Level 2 plus the simplex matrix, :math:`x^{(i)}`, and value of the objective function at each vertex of the simplex.

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
        
            bool success = optim::nm(x, sphere_fn, nullptr);
        
            if (success) {
                std::cout << "nm: sphere test completed successfully." << "\n";
            } else {
                std::cout << "nm: sphere test completed unsuccessfully." << "\n";
            }
        
            arma::cout << "nm: solution to sphere test:\n" << x << arma::endl;
        
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
        
            bool success = optim::nm(x, sphere_fn, nullptr);
        
            if (success) {
                std::cout << "nm: sphere test completed successfully." << "\n";
            } else {
                std::cout << "nm: sphere test completed unsuccessfully." << "\n";
            }
        
            std::cout << "nm: solution to sphere test:\n" << x << std::endl;
        
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
        
            bool success_2 = optim::nm(x, booth_fn, nullptr);
        
            if (success_2) {
                std::cout << "nm: Booth test completed successfully." << "\n";
            } else {
                std::cout << "nm: Booth test completed unsuccessfully." << "\n";
            }
        
            arma::cout << "nm: solution to Booth test:\n" << x_2 << arma::endl;
        
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
        
            bool success_2 = optim::nm(x, booth_fn, nullptr);
        
            if (success_2) {
                std::cout << "nm: Booth test completed successfully." << "\n";
            } else {
                std::cout << "nm: Booth test completed unsuccessfully." << "\n";
            }
        
            std::cout << "nm: solution to Booth test:\n" << x_2 << std::endl;
        
            return 0;
        }

----
