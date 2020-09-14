.. Copyright (c) 2016-2020 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Gradient Descent
================

**Table of contents**

.. contents:: :local:

----

Algorithm Description
---------------------

The Gradient Descent (GD) algorithm is used solve to optimization problems of the form

.. math::

    \min_{x \in X} f(x)

where :math:`f` is convex and (at least) once differentiable. 

The updating rule for gradient descent is described below. Let :math:`x^{(i)}` denote the candidate solution vector at stage :math:`i` of the algorithm.

1. Compute the descent direction :math:`d^{(i)}` using one of the methods described below.

3. Update the candidate solution vector using:

    .. math::

        x^{(i+1)} = x^{(i)} - d^{(i)}

The algorithm stops when one of the following conditions are ``true``:

  1. the norm of the gradient vector, :math:`\| \nabla f \|`, is less than ``grad_err_tol``;

  2. the relative change between :math:`x^{(i+1)}` and :math:`x^{(i)}` is less than ``rel_sol_change_tol``;

  3. the total number of iterations exceeds ``iter_max``.

----

Gradient Descent Rule
~~~~~~~~~~~~~~~~~~~~~

- ``gd_settings.method = 0`` Vanilla GD:

  .. math::

    d^{(i)} = \alpha \times [ \nabla_x f( x^{(i)} ) ]

  where :math:`\alpha`, the step size (also known the learning rate), is set by ``par_step_size``.

- ``gd_settings.method = 1`` GD with **momentum**:

  .. math::

    d^{(i)} = \mu \times d^{(i-1)} + \alpha \times [ \nabla_x f( x^{(i)} ) ]

  where :math:`\mu`, the momentum parameter, is set by ``par_momentum``.

- ``gd_settings.method = 2`` Nesterov accelerated gradient descent (**NAG**)

  .. math::

    d^{(i)} = \mu \times d^{(i-1)} + \alpha \times \nabla f( x^{(i)} -  \mu \times d^{(i-1)})

- ``gd_settings.method = 3`` **AdaGrad**:

  .. math::

    \begin{aligned}
    d^{(i)} &= [ \nabla_x f( x^{(i)} ) ] \odot \dfrac{1}{\sqrt{v^{(i)}} + \epsilon} \\
    v^{(i)} &= v^{(i-1)} + [ \nabla_x f( x^{(i)} ) ] \odot [ \nabla_x f( x^{(i)} ) ]
    \end{aligned}

- ``gd_settings.method = 4`` **RMSProp**:

  .. math::

    \begin{aligned}
    d^{(i)} &= [ \nabla_x f( x^{(i)} ) ] \odot \dfrac{1}{\sqrt{v^{(i)}} + \epsilon} \\
    v^{(i)} &= \rho \times v^{(i-1)} + (1-\rho) \times [ \nabla_x f( x^{(i)} ) ] \odot [ \nabla_x f( x^{(i)} ) ]
    \end{aligned}

- ``gd_settings.method = 5`` **AdaDelta**:

  .. math::

    \begin{aligned}
    d^{(i)} &= [ \nabla_x f( x^{(i)} ) ] \odot \dfrac{\sqrt{m^{(i)}} + \epsilon}{\sqrt{v^{(i)}} + \epsilon} \\
    m^{(i)} &= \rho \times m^{(i-1)} + (1-\rho) \times [ d^{(i-1)} ] \odot [ d^{(i-1)} ] \\
    v^{(i)} &= \rho \times v^{(i-1)} + (1-\rho) \times [ \nabla_x f( x^{(i)} ) ] \odot [ \nabla_x f( x^{(i)} ) ]
    \end{aligned}

- ``gd_settings.method = 6`` **Adam** (adaptive moment estimation) and **AdaMax**.

  .. math::

    \begin{aligned}
    m^{(i)} &= \beta_1 \times m^{(i-1)} + (1-\beta_1) \times [ \nabla_x f( x^{(i-1)} ) ] \\
    v^{(i)} &= \beta_2 \times v^{(i-1)} + (1-\beta_2) \times [ \nabla_x f( x^{(i)} ) ] \odot [ \nabla_x f( x^{(i)} ) ] \\
    & \ \ \ \ \ \ \hat{m} = \dfrac{m^{(i)}}{1 - \beta_1^i}, \ \ \hat{v} = \dfrac{v^{(i)}}{1 - \beta_2^i}
    \end{aligned}

  where :math:`m^{(0)} = \mathbf{0}`, and :math:`\beta_1` and :math:`\beta_2` are set by ``par_adam_beta_1`` and ``par_adam_beta_2``, respectively.

  - If ``ada_max = false``, then the descent direction is computed as

    .. math::

      d^{(i)} = \alpha \times \dfrac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}

  - If ``ada_max = true``, then the updating rule for :math:`v^{(i)}` is no longer based on the :math:`L_2` norm; instead

    .. math::

      v^{(i)} = \max \left\{ \beta_2 \times v^{(i-1)}, | \nabla_x f( x^{(i)} ) | \right\}

    The descent direction is computed using

    .. math::

      d^{(i)} = \alpha \times \dfrac{\hat{m}}{ v^{(i)} + \epsilon}

- ``gd_settings.method = 7`` **Nadam** (adaptive moment estimation) and **NadaMax**

  .. math::

    \begin{aligned}
    m^{(i)} &= \beta_1 \times m^{(i-1)} + (1-\beta_1) \times [ \nabla_x f( x^{(i-1)} ) ] \\
    v^{(i)} &= \beta_2 \times v^{(i-1)} + (1-\beta_2) \times [ \nabla_x f( x^{(i)} ) ] \odot [ \nabla_x f( x^{(i)} ) ] \\
    & \ \ \hat{m} = \dfrac{m^{(i)}}{1 - \beta_1^i}, \ \ \hat{v} = \dfrac{v^{(i)}}{1 - \beta_2^i}, \ \ \hat{g} = \dfrac{ \nabla_x f(x^{(i)}) }{1 - \beta_1^i}
    \end{aligned}

  where :math:`m^{(0)} = \mathbf{0}`, and :math:`\beta_1` and :math:`\beta_2` are set by ``par_adam_beta_1`` and ``par_adam_beta_2``, respectively.

  - If ``ada_max = false``, then the descent direction is computed as

    .. math::

      d^{(i)} = \alpha \times [ \nabla_x f( x^{(i)} ) ] \odot \dfrac{\beta_1 \hat{m} + (1 - \beta_1) \hat{g} }{\sqrt{\hat{v}} + \epsilon}

  - If ``ada_max = true``, then the updating rule for :math:`v^{(i)}` is no longer based on the :math:`L_2` norm; instead

    .. math::

      v^{(i)} = \max \left\{ \beta_2 \times v^{(i-1)}, | \nabla_x f( x^{(i)} ) | \right\}

    The descent direction is computed using

    .. math::

      d^{(i)} = \alpha \times [ \nabla_x f( x^{(i)} ) ] \odot \dfrac{\beta_1 \hat{m} + (1 - \beta_1) \hat{g} }{v^{(i)} + \epsilon}

----

Function Declarations
---------------------

.. _gd-func-ref1:
.. doxygenfunction:: gd(Vec_t&, std::function<doubleconst Vec_t &vals_inp, Vec_t *grad_out, void *opt_data>, void *)
   :project: optimlib

.. _gd-func-ref2:
.. doxygenfunction:: gd(Vec_t&, std::function<doubleconst Vec_t &vals_inp, Vec_t *grad_out, void *opt_data>, void *, algo_settings_t&)
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

In addition to these:

- ``int print_level``: Set the level of detail for printing updates on optimization progress.

  - Level ``0``: Nothing (default).

  - Level ``1``: Print the current iteration count and error values.

  - Level ``2``: Level 1 plus the current candidate solution values, :math:`x^{(i+1)}`.

  - Level ``3``: Level 2 plus the direction vector, :math:`d^{(i)}`, and the gradient vector, :math:`\nabla_x f(x^{(i+1)})`.

  - Level ``4``: Level 3 plus information about the chosen gradient descent rule.

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
        
            bool success = optim::gd(x, sphere_fn, nullptr);
        
            if (success) {
                std::cout << "gd: sphere test completed successfully." << "\n";
            } else {
                std::cout << "gd: sphere test completed unsuccessfully." << "\n";
            }
        
            arma::cout << "gd: solution to sphere test:\n" << x << arma::endl;
        
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
        
            bool success = optim::gd(x, sphere_fn, nullptr);
        
            if (success) {
                std::cout << "gd: sphere test completed successfully." << "\n";
            } else {
                std::cout << "gd: sphere test completed unsuccessfully." << "\n";
            }
        
            std::cout << "gd: solution to sphere test:\n" << x << std::endl;
        
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
        
            bool success_2 = optim::gd(x, booth_fn, nullptr);
        
            if (success_2) {
                std::cout << "gd: Booth test completed successfully." << "\n";
            } else {
                std::cout << "gd: Booth test completed unsuccessfully." << "\n";
            }
        
            arma::cout << "gd: solution to Booth test:\n" << x_2 << arma::endl;
        
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
        
            bool success_2 = optim::gd(x, booth_fn, nullptr);
        
            if (success_2) {
                std::cout << "gd: Booth test completed successfully." << "\n";
            } else {
                std::cout << "gd: Booth test completed unsuccessfully." << "\n";
            }
        
            std::cout << "gd: solution to Booth test:\n" << x_2 << std::endl;
        
            return 0;
        }

----
