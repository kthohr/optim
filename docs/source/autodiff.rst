.. Copyright (c) 2016-2020 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Automatic Differentiation
=========================

Gradient-based methods in OptimLib (such as BFGS and gradient descent) require the user-provided function to return a gradient vector at each evaluation. This is best achieved by knowing the gradient in closed form, but OptimLib also provides **experimental support** for automatic differentiation via the `autodiff library <https://autodiff.github.io>`_. 

Requirements: an Eigen-based build of OptimLib, a copy of the ``autodiff`` headers, and a C++17 compatible compiler.

----

Example
-------

The example below uses forward-mode automatic differentiation to compute the gradient of the sphere function, and the BFGS algorithm is used to find input values that minimize the autodiff-enabled function.

.. code:: cpp

    #define OPTIM_ENABLE_EIGEN_WRAPPERS
    #include "optim.hpp"

    #include <autodiff/forward.hpp>
    #include <autodiff/forward/eigen.hpp>

    //

    autodiff::dual 
    opt_fnd(const autodiff::VectorXdual& x)
    {
        return x.cwiseProduct(x).sum();
    }

    double
    opt_fn(const Eigen::VectorXd& x, Eigen::VectorXd* grad_out, void* opt_data)
    {
        autodiff::dual u;
        autodiff::VectorXdual xd = x.eval();

        if (grad_out) {
            Eigen::VectorXd grad_tmp = autodiff::forward::gradient(opt_fnd, autodiff::wrt(xd), autodiff::forward::at(xd), u);

            *grad_out = grad_tmp;
        } else {
            u = opt_fnd(xd);
        }

        return u.val;
    }

    int main()
    {
        Eigen::VectorXd x(5);
        x << 1, 2, 3, 4, 5;

        bool success = optim::bfgs(x, opt_fn, nullptr);

        if (success) {
            std::cout << "bfgs: autodiff test completed successfully." << std::endl;
        } else {
            std::cout << "bfgs: autodiff test completed unsuccessfully." << std::endl;
        }

        std::cout << "solution: x = \n" << x << std::endl;

        return 0;
    }


This example can be compiled using:

.. code:: bash

    g++ -Wall -std=c++17 -O3 -march=native -ffp-contract=fast -I/path/to/eigen -I/path/to/autodiff -I/path/to/optim/include optim_autodiff_ex.cpp -o optim_autodiff_ex.out -L/path/to/optim/lib -loptim


----
