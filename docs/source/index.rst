.. Copyright (c) 2016-2022 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Introduction
============

OptimLib is a lightweight C++ library of numerical optimization methods for nonlinear functions.

Features:

- A C++11 library of local and global optimization algorithms, as well as root finding techniques.

- Derivative-free optimization using advanced, parallelized metaheuristic methods.

- Constrained optimization routines to handle simple box constraints, as well as systems of nonlinear constraints.

- For fast and efficient matrix-based computation, OptimLib supports the following templated linear algebra libraries:

  - `Armadillo <http://arma.sourceforge.net/>`_ 

  - `Eigen <http://eigen.tuxfamily.org/index.php>`_ (version >= 3.4.0)

- Automatic differentiation functionality is available through use of the `Autodiff library <https://autodiff.github.io>`_

- OpenMP-accelerated algorithms for parallel computation. 

- Straightforward linking with parallelized BLAS libraries, such as `OpenBLAS <https://github.com/xianyi/OpenBLAS>`_.

- Available as a header-only library, or as a compiled shared library.

- Released under a permissive, non-GPL license.

Author: Keith O'Hara

License: Apache Version 2.0

----

Installation
------------

The library can be installed on Unix-alike systems via the standard ``./configure && make`` method.

See the installation page for :ref:`detailed instructions <installation>`.

Algorithms
----------

A list of currently available algorithms includes:

* Broyden's Method (for root finding)
* Newton's method, BFGS, and L-BFGS
* Gradient descent: basic, momentum, Adam, AdaMax, Nadam, NadaMax, and more
* Nonlinear Conjugate Gradient
* Nelder-Mead
* Differential Evolution (DE)
* Particle Swarm Optimization (PSO)

----

Contents
--------

.. toctree::
   :caption: Guide
   :maxdepth: 2
   
   installation
   examples_and_tests
   settings
   autodiff

.. toctree::
   :caption: Algorithms
   :maxdepth: 2
   
   api/convex_algo_index
   api/simplex_algo_index
   api/metaheuristic_algo_index
   api/constrained_algo_index
   api/root_finding_algo_index

.. toctree::
   :caption: Appendix
   :maxdepth: 2
   
   box_constraints
   line_search
   test_functions
