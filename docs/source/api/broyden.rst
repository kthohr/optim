.. Copyright (c) 2016-2022 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Broyden
=======

**Table of contents**

.. contents:: :local:

----

Algorithm Description
---------------------

Broyden's method is an algorithm for solving systems of nonlinear equations,

.. math::

    F(x^{(*)}) = \mathbf{0}

where :math:`F : \mathbb{R}^n \to \mathbb{R}^m` is convex and differentiable. The algorithm uses an approximation to the Jacobian. 

The updating rule for Broyden's method is described below. Let :math:`x^{(i)}` denote the function input values at stage :math:`i` of the algorithm.

1. Compute the descent direction using:

    .. math::

        d^{(i)} = - B^{(i)} F(x^{(i)})

2. Update the candidate solution vector using:

.. math::

    x^{(i+1)} = x^{(i)} + d^{(i)}

3. Update the approximate inverse Jacobian matrix, :math:`B`, using:

    .. math::

        B^{(i+1)} = B^{(i)} + \frac{1}{[y^{(i+1)}]^\top y^{(i+1)}} (s^{(i+1)} - B^{(i)} y^{(i+1)}) [y^{(i+1)}]^\top

  where

    .. math::

        \begin{aligned}
            s^{(i)} &:= x^{(i)} - x^{(i-1)} \\
            y^{(i)} &:= F(x^{(i)}) - F(x^{(i-1)})
        \end{aligned}


The algorithm stops when at least one of the following conditions are met:

  1. :math:`\| F \|` is less than ``rel_objfn_change_tol``.

  2. the relative change between :math:`x^{(i+1)}` and :math:`x^{(i)}` is less than ``rel_sol_change_tol``;

  3. the total number of iterations exceeds ``iter_max``.


----

Function Declarations
---------------------

.. _broyden-func-ref1:
.. doxygenfunction:: broyden(ColVec_t&, std::function<Vec_tconst ColVec_t &vals_inp, void *opt_data>, void *)
   :project: optimlib

.. _broyden-func-ref2:
.. doxygenfunction:: broyden(ColVec_t&, std::function<Vec_tconst ColVec_t &vals_inp, void *opt_data>, void *, algo_settings_t&)
   :project: optimlib

.. _broyden-func-ref3:
.. doxygenfunction:: broyden(ColVec_t&, std::function<Vec_tconst ColVec_t &vals_inp, void *opt_data>, void *, std::function<Mat_tconst ColVec_t &vals_inp, void *jacob_data>, void *)
   :project: optimlib

.. _broyden-func-ref4:
.. doxygenfunction:: broyden(ColVec_t&, std::function<Vec_tconst ColVec_t &vals_inp, void *opt_data>, void *, std::function<Mat_tconst ColVec_t &vals_inp, void *jacob_data>, void *, algo_settings_t&)
   :project: optimlib

----

Optimization Control Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basic control parameters are:

- ``fp_t rel_objfn_change_tol``: the error tolerance value controlling how small :math:`\| F \|` should be before 'convergence' is declared.

- ``fp_t rel_sol_change_tol``: the error tolerance value controlling how small the proportional change in the solution vector should be before 'convergence' is declared.

  The relative change is computed using:

    .. math::

        \left\| \dfrac{x^{(i)} - x^{(i-1)}}{ |x^{(i-1)}| + \epsilon } \right\|_1

  where :math:`\epsilon` is a small number added for numerical stability.

- ``size_t iter_max``: the maximum number of iterations/updates before the algorithm exits.

- ``bool vals_bound``: whether the search space of the algorithm is bounded. If ``true``, then

  - ``ColVec_t lower_bounds``: defines the lower bounds of the search space.

  - ``ColVec_t upper_bounds``: defines the upper bounds of the search space.

In addition to these:

- ``int print_level``: Set the level of detail for printing updates on optimization progress.

  - Level ``0``: Nothing (default).

  - Level ``1``: Print the current iteration count and error values.

  - Level ``2``: Level 1 plus the current candidate solution values, :math:`x^{(i+1)}`.

  - Level ``3``: Level 2 plus the direction vector, :math:`d^{(i)}`, and the function values, :math:`F(x^{(i+1)})`.

  - Level ``4``: Level 3 plus the components used to update the approximate inverse Jacobian matrix: :math:`s^{(i+1)}`, :math:`y^{(i+1)}`, and :math:`B^{(i+1)}`.

----

Examples
--------

Example 1
~~~~~~~~~

.. math::

    F(\mathbf{x}) = \begin{bmatrix} \exp(-\exp(-(x_1+x_2))) - x_2(1+x_1^2) \\ x_1\cos(x_2) + x_2\sin(x_1) - 0.5 \end{bmatrix}


Code to run this example is given below.

.. toggle-header::
    :header: **Armadillo (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_ARMA_WRAPPERS
        #include "optim.hpp"
        
        inline
        arma::vec
        zeros_test_objfn_1(const arma::vec& vals_inp, void* opt_data)
        {
            double x_1 = vals_inp(0);
            double x_2 = vals_inp(1);

            //

            arma::vec ret(2);

            ret(0) = std::exp(-std::exp(-(x_1+x_2))) - x_2*(1 + std::pow(x_1,2));
            ret(1) = x_1*std::cos(x_2) + x_2*std::sin(x_1) - 0.5;
            
            //

            return ret;
        }

        inline
        arma::mat
        zeros_test_jacob_1(const arma::vec& vals_inp, void* opt_data)
        {
            double x_1 = vals_inp(0);
            double x_2 = vals_inp(1);

            //

            arma::mat ret(2,2);

            ret(0,0) = std::exp(-std::exp(-(x_1+x_2))-(x_1+x_2)) - 2*x_1*x_1;
            ret(0,1) = std::exp(-std::exp(-(x_1+x_2))-(x_1+x_2)) - x_1*x_1 - 1.0;
            ret(1,0) = std::cos(x_2) + x_2*std::cos(x_1);
            ret(1,1) = -x_1*std::sin(x_2) + std::cos(x_1);

            //
            
            return ret;
        }
        
        int main()
        {
            arma::vec x = arma::zeros(2,1); // initial values (0,0)
        
            bool success = optim::broyden(x, zeros_test_objfn_1, nullptr);
        
            if (success) {
                std::cout << "broyden: test_1 completed successfully." << "\n";
            } else {
                std::cout << "broyden: test_1 completed unsuccessfully." << "\n";
            }
        
            arma::cout << "broyden: solution to test_1:\n" << x << arma::endl;

            //

            x = arma::zeros(2,1);
        
            success = optim::broyden(x, zeros_test_objfn_1, nullptr, zeros_test_jacob_1, nullptr);
        
            if (success) {
                std::cout << "broyden with jacobian: test_1 completed successfully." << "\n";
            } else {
                std::cout << "broyden with jacobian: test_1 completed unsuccessfully." << "\n";
            }
        
            arma::cout << "broyden with jacobian: solution to test_1:\n" << x << arma::endl;

            //
        
            return 0;
        }

.. toggle-header::
    :header: **Eigen (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_EIGEN_WRAPPERS
        #include "optim.hpp"

        inline
        Eigen::VectorXd
        zeros_test_objfn_1(const Eigen::VectorXd& vals_inp, void* opt_data)
        {
            double x_1 = vals_inp(0);
            double x_2 = vals_inp(1);

            //

            Eigen::VectorXd ret(2);

            ret(0) = std::exp(-std::exp(-(x_1+x_2))) - x_2*(1 + std::pow(x_1,2));
            ret(1) = x_1*std::cos(x_2) + x_2*std::sin(x_1) - 0.5;
            
            //

            return ret;
        }

        inline
        Eigen::MatrixXd
        zeros_test_jacob_1(const Eigen::VectorXd& vals_inp, void* opt_data)
        {
            double x_1 = vals_inp(0);
            double x_2 = vals_inp(1);

            //

            Eigen::MatrixXd ret(2,2);

            ret(0,0) = std::exp(-std::exp(-(x_1+x_2))-(x_1+x_2)) - 2*x_1*x_1;
            ret(0,1) = std::exp(-std::exp(-(x_1+x_2))-(x_1+x_2)) - x_1*x_1 - 1.0;
            ret(1,0) = std::cos(x_2) + x_2*std::cos(x_1);
            ret(1,1) = -x_1*std::sin(x_2) + std::cos(x_1);

            //
            
            return ret;
        }
        
        int main()
        {
            Eigen::VectorXd x = Eigen::VectorXd::Zero(2); // initial values (0,0)
        
            bool success = optim::broyden(x, zeros_test_objfn_1, nullptr);
        
            if (success) {
                std::cout << "broyden: test_1 completed successfully." << "\n";
            } else {
                std::cout << "broyden: test_1 completed unsuccessfully." << "\n";
            }
        
            std::cout << "broyden: solution to test_1:\n" << x << std::endl;

            //

            x = Eigen::VectorXd::Zero(2);
        
            success = optim::broyden(x, zeros_test_objfn_1, nullptr, zeros_test_jacob_1, nullptr);
        
            if (success) {
                std::cout << "broyden with jacobian: test_1 completed successfully." << "\n";
            } else {
                std::cout << "broyden with jacobian: test_1 completed unsuccessfully." << "\n";
            }
        
            std::cout << "broyden with jacobian: solution to test_1:\n" << x << std::endl;

            //
        
            return 0;
        }

----

Example 2
~~~~~~~~~

.. math::

    F(\mathbf{x}) = \begin{bmatrix} 2x_1 - x_2 - \exp(-x_1) \\ - x_1 + 2x_2 - \exp(-x_2) \end{bmatrix}


Code to run this example is given below.

.. toggle-header::
    :header: **Armadillo (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_ARMA_WRAPPERS
        #include "optim.hpp"
        
        inline
        arma::vec
        zeros_test_objfn_2(const arma::vec& vals_inp, void* opt_data)
        {
            double x_1 = vals_inp(0);
            double x_2 = vals_inp(1);

            //

            arma::vec ret(2);

            ret(0) =   2*x_1 - x_2   - std::exp(-x_1);
            ret(1) = - x_1   + 2*x_2 - std::exp(-x_2);
            
            //

            return ret;
        }

        inline
        arma::mat
        zeros_test_jacob_2(const arma::vec& vals_inp, void* opt_data)
        {
            double x_1 = vals_inp(0);
            double x_2 = vals_inp(1);

            //

            arma::mat ret(2,2);

            ret(0,0) = 2 + std::exp(-x_1);
            ret(0,1) = - 1.0;
            ret(1,0) = - 1.0;
            ret(1,1) = 2 + std::exp(-x_2);

            //
            
            return ret;
        }
        
        int main()
        {
            arma::vec x = arma::zeros(2,1); // initial values (0,0)
        
            bool success = optim::broyden(x, zeros_test_objfn_2, nullptr);
        
            if (success) {
                std::cout << "broyden: test_2 completed successfully." << "\n";
            } else {
                std::cout << "broyden: test_2 completed unsuccessfully." << "\n";
            }
        
            arma::cout << "broyden: solution to test_2:\n" << x << arma::endl;

            //

            x = arma::zeros(2,1);
        
            success = optim::broyden(x, zeros_test_objfn_2, nullptr, zeros_test_jacob_2, nullptr);
        
            if (success) {
                std::cout << "broyden with jacobian: test_2 completed successfully." << "\n";
            } else {
                std::cout << "broyden with jacobian: test_2 completed unsuccessfully." << "\n";
            }
        
            arma::cout << "broyden with jacobian: solution to test_2:\n" << x << arma::endl;

            //
        
            return 0;
        }

.. toggle-header::
    :header: **Eigen (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_EIGEN_WRAPPERS
        #include "optim.hpp"

        inline
        Eigen::VectorXd
        zeros_test_objfn_2(const Eigen::VectorXd& vals_inp, void* opt_data)
        {
            double x_1 = vals_inp(0);
            double x_2 = vals_inp(1);

            //

            Eigen::VectorXd ret(2);

            ret(0) =   2*x_1 - x_2   - std::exp(-x_1);
            ret(1) = - x_1   + 2*x_2 - std::exp(-x_2);
            
            //

            return ret;
        }

        inline
        Eigen::MatrixXd
        zeros_test_jacob_2(const Eigen::VectorXd& vals_inp, void* opt_data)
        {
            double x_1 = vals_inp(0);
            double x_2 = vals_inp(1);

            //

            Eigen::MatrixXd ret(2,2);

            ret(0,0) = 2 + std::exp(-x_1);
            ret(0,1) = - 1.0;
            ret(1,0) = - 1.0;
            ret(1,1) = 2 + std::exp(-x_2);

            //
            
            return ret;
        }
        
        int main()
        {
            Eigen::VectorXd x = Eigen::VectorXd::Zero(2); // initial values (0,0)
        
            bool success = optim::broyden(x, zeros_test_objfn_2, nullptr);
        
            if (success) {
                std::cout << "broyden: test_2 completed successfully." << "\n";
            } else {
                std::cout << "broyden: test_2 completed unsuccessfully." << "\n";
            }
        
            std::cout << "broyden: solution to test_2:\n" << x << std::endl;

            //

            x = Eigen::VectorXd::Zero(2);
        
            success = optim::broyden(x, zeros_test_objfn_2, nullptr, zeros_test_jacob_2, nullptr);
        
            if (success) {
                std::cout << "broyden with jacobian: test_2 completed successfully." << "\n";
            } else {
                std::cout << "broyden with jacobian: test_2 completed unsuccessfully." << "\n";
            }
        
            std::cout << "broyden with jacobian: solution to test_2:\n" << x << std::endl;

            //
        
            return 0;
        }

----
