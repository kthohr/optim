.. Copyright (c) 2016-2022 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Examples and Tests
==================

.. contents:: :local:

----

API
---

The OptimLib API follows a relatively simple convention, with most algorithms called using the following syntax:

.. code::
    
    algorithm_id(<initial/final values>, <objective function>, <objective function data>);

The function inputs, in order, are:

- A writable vector of initial values to define the starting point of the algorithm, where, in the event of successful completion of the algorithm, the initial values will be overwritten by the latest candidate solution vector.
- The ``objective function`` is a user-defined function to be minimized, or zeroed-out in the case of root finding methods.
- The final input is optional; it is any object that contains additional parameters necessary to evaluate the objective function.

For example, the BFGS algorithm is called using

.. code:: cpp

    bfgs(ColVec_t& init_out_vals, std::function<double (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, void* opt_data);


----

Example
-------

The code below uses Differential Evolution to search for the minimum of the :ref:`Ackley function <ackley_fn>`.

.. code:: cpp

    #define OPTIM_ENABLE_ARMA_WRAPPERS
    #include "optim.hpp"

    // Ackley function

    double ackley_fn(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
    {
        const double x = vals_inp(0);
        const double y = vals_inp(1);
        const double pi = arma::datum::pi;

        double obj_val = -20*std::exp( -0.2*std::sqrt(0.5*(x*x + y*y)) ) - std::exp( 0.5*(std::cos(2*pi*x) + std::cos(2*pi*y)) ) + 22.718282L;

        //

        return obj_val;
    }

    int main()
    {
        // initial values:
        arma::vec x = arma::ones(2,1) + 1.0; // (2,2)

        //

        std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

        bool success = optim::de(x,ackley_fn,nullptr);

        std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        if (success) {
            std::cout << "de: Ackley test completed successfully.\n"
                    << "elapsed time: " << elapsed_seconds.count() << "s\n";
        } else {
            std::cout << "de: Ackley test completed unsuccessfully." << std::endl;
        }

        arma::cout << "\nde: solution to Ackley test:\n" << x << arma::endl;

        return 0;
    }

On x86-based computers, this example can be compiled using:

.. code:: bash

    g++ -Wall -std=c++11 -O3 -march=native -ffp-contract=fast -I/path/to/armadillo -I/path/to/optim/include optim_de_ex.cpp -o optim_de_ex.out -L/path/to/optim/lib -loptim


----

Test suite
----------

You can build the test suite as follows:

.. code:: bash

    # compile tests
    cd ./tests
    ./setup
    cd ./unconstrained
    ./configure -l arma
    make
    ./bfgs.test
