.. Copyright (c) 2016-2020 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Newton's Method
===============

**Table of contents**

.. contents:: :local:

----

Algorithm Description
---------------------

Newton's method is used solve to convex optimization problems of the form

.. math::

    \min_{x \in X} f(x)

where :math:`f` is convex and twice differentiable. The algorithm requires both the gradient and Hessian to be known.

The updating rule for Newton's method is described below. Let :math:`x^{(i)}` denote the candidate solution vector at stage :math:`i` of the algorithm.

1. Compute the descent direction using:

    .. math::

        d^{(i)} = - [H(x^{(i)})]^{-1} [\nabla_x f(x^{(i)})]

2. Compute the optimal step size using line search:

    .. math::

        \alpha^{(i)} = \arg \min_{\alpha} f(x^{(i)} + \alpha d^{(i)})

3. Update the candidate solution vector using:

.. math::

    x^{(i+1)} = x^{(i)} + \alpha^{(i)} d^{(i)}


The algorithm stops when one of the following conditions are ``true``:

  1. the norm of the gradient vector, :math:`\| \nabla f \|`, is less than ``grad_err_tol``;

  2. the relative change between :math:`x^{(i+1)}` and :math:`x^{(i)}` is less than ``rel_sol_change_tol``;

  3. the total number of iterations exceeds ``iter_max``.

----

Function Declarations
---------------------

.. _newton-func-ref1:
.. doxygenfunction:: newton(Vec_t&, std::function<doubleconst Vec_t &vals_inp, Vec_t *grad_out, Mat_t *hess_out, void *opt_data>, void *)
   :project: optimlib

.. _newton-func-ref2:
.. doxygenfunction:: newton(Vec_t&, std::function<doubleconst Vec_t &vals_inp, Vec_t *grad_out, Mat_t *hess_out, void *opt_data>, void *, algo_settings_t&)
   :project: optimlib

----

Optimization Control Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basic control parameters are:

- ``double grad_err_tol``: the error tolerance value controlling how small the L2 norm of the gradient vector :math:`\| \nabla f \|` should be before 'convergence' is declared.

- ``double rel_sol_change_tol``: the error tolerance value controlling how small the proportional change in the solution vector should be before 'convergence' is declared.

  The relative change is computed using:

    .. math::

        \left\| \dfrac{x^{(i)} - x^{(i-1)}}{ |x^{(i-1)}| + \epsilon } \right\|_1

- ``size_t iter_max``: the maximum number of iterations/updates before the algorithm exits.

In addition to these:

- ``int print_level``: Set the level of detail for printing updates on optimization progress.

  - Level ``0``: Nothing (default).

  - Level ``1``: Print the iteration count and current error values.

  - Level ``2``: Level 1 plus the current candidate solution values, :math:`x^{(i+1)}`.

  - Level ``3``: Level 2 plus the direction vector, :math:`d^{(i)}`, and the gradient vector, :math:`\nabla_x f(x^{(i+1)})`.

  - Level ``4``: Level 3 plus the Hessian matrix, :math:`H(x^{(i)})`.

----

Examples
--------

Example 1
~~~~~~~~~

Code to run this example is given below.

.. toggle-header::
    :header: **Armadillo (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_ARMA_WRAPPERS
        #include "optim.hpp"
        
        inline
        double
        unconstr_test_fn_1_whess(const arma::vec& vals_inp, arma::vec* grad_out, arma::mat* hess_out, void* opt_data)
        {
            const double x_1 = vals_inp(0);
            const double x_2 = vals_inp(1);

            double obj_val = 3*x_1*x_1 + 2*x_1*x_2 + x_2*x_2 - 4*x_1 + 5*x_2;

            if (grad_out) {
                (*grad_out)(0) = 6*x_1 + 2*x_2 - 4;
                (*grad_out)(1) = 2*x_1 + 2*x_2 + 5;
            }

            if (hess_out) {
                (*hess_out)(0,0) = 6.0;
                (*hess_out)(0,1) = 2.0;
                (*hess_out)(1,0) = 2.0;
                (*hess_out)(1,1) = 2.0;
            }

            //
            
            return obj_val;
        }
        
        int main()
        {
            arma::vec x = arma::zeros(2,1);
        
            bool success = optim::newton(x, unconstr_test_fn_1_whess, nullptr);
        
            if (success) {
                std::cout << "newton: test completed successfully." << "\n";
            } else {
                std::cout << "newton: test completed unsuccessfully." << "\n";
            }
        
            arma::cout << "newton: solution to test:\n" << x << arma::endl;
        
            return 0;
        }

.. toggle-header::
    :header: **Eigen (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_EIGEN_WRAPPERS
        #include "optim.hpp"
        
        inline
        double
        unconstr_test_fn_1_whess(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, Eigen::MatrixXd* hess_out, void* opt_data)
        {
            const double x_1 = vals_inp(0);
            const double x_2 = vals_inp(1);

            double obj_val = 3*x_1*x_1 + 2*x_1*x_2 + x_2*x_2 - 4*x_1 + 5*x_2;

            if (grad_out) {
                (*grad_out)(0) = 6*x_1 + 2*x_2 - 4;
                (*grad_out)(1) = 2*x_1 + 2*x_2 + 5;
            }

            if (hess_out) {
                (*hess_out)(0,0) = 6.0;
                (*hess_out)(0,1) = 2.0;
                (*hess_out)(1,0) = 2.0;
                (*hess_out)(1,1) = 2.0;
            }

            //
            
            return obj_val;
        }
        
        int main()
        {
            Eigen::VectorXd x = Eigen::VectorXd::Zero(2); // initial values (1,1,...,1)
        
            bool success = optim::newton(x, unconstr_test_fn_1_whess, nullptr);
        
            if (success) {
                std::cout << "newton: test completed successfully." << "\n";
            } else {
                std::cout << "newton: test completed unsuccessfully." << "\n";
            }
        
            std::cout << "newton: solution to test:\n" << x << std::endl;
        
            return 0;
        }

----
