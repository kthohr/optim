.. Copyright (c) 2016-2022 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

.. _installation:

Installation
============

OptimLib is available as a compiled shared library, or as header-only library, for Unix-alike systems only (e.g., popular Linux-based distros, as well as macOS). Note that use of this library with Windows-based systems, with or without MSVC, **is not supported**.


Requirements
------------

OptimLib requires either the Armadillo or Eigen C++ linear algebra libraries. (Note that Eigen version 3.4.0 requires a C++14-compatible compiler.)

The following options should be declared **before** including the OptimLib header files. 

- OpenMP functionality is enabled by default if the ``_OPENMP`` macro is detected (e.g., by invoking ``-fopenmp`` with GCC or Clang). 

  - To explicitly enable OpenMP features, use:

  .. code:: cpp

    #define OPTIM_USE_OPENMP

  - To explicitly disable OpenMP functionality, use:

  .. code:: cpp

    #define OPTIM_DONT_USE_OPENMP

- To use OptimLib with Armadillo or Eigen:

  .. code:: cpp

    #define OPTIM_ENABLE_ARMA_WRAPPERS
    #define OPTIM_ENABLE_EIGEN_WRAPPERS

  Example:

  .. code:: cpp

    #define OPTIM_ENABLE_EIGEN_WRAPPERS
    #include "optim.hpp"

- To use OptimLib with RcppArmadillo or RcppEigen:

  .. code:: cpp

    #define OPTIM_USE_RCPP_ARMADILLO
    #define OPTIM_USE_RCPP_EIGEN

  Example:

  .. code:: cpp

    #define OPTIM_USE_RCPP_EIGEN
    #include "optim.hpp"


----

Installation Method 1: Shared Library
-------------------------------------

The library can be installed on Unix-alike systems via the standard ``./configure && make`` method.

The primary configuration options can be displayed by calling ``./configure -h``, which results in:

.. code:: bash

    $ ./configure -h

    OptimLib Configuration

    Main options:
    -c    Code coverage build
            (default: disabled)
    -d    Developmental build
            (default: disabled)
    -f    Floating-point number type
            (default: double)
    -g    Debugging build (optimization flags set to -O0 -g)
            (default: disabled)
    -h    Print help
    -i    Install path (default: current directory)
            Example: /usr/local
    -l    Choice of linear algebra library
            Examples: -l arma or -l eigen
    -m    Specify the BLAS and Lapack libraries to link against
            Examples: -m "-lopenblas" or -m "-framework Accelerate"
    -o    Compiler optimization options
            (default: -O3 -march=native -ffp-contract=fast -flto -DARMA_NO_DEBUG)
    -p    Enable OpenMP parallelization features
            (default: disabled)

    Special options:
    --header-only-version    Generate a header-only version of OptimLib

If choosing a shared library build, set (one) of the following environment variables *before* running `configure`:

.. code:: bash

    export ARMA_INCLUDE_PATH=/path/to/armadillo
    export EIGEN_INCLUDE_PATH=/path/to/eigen

Then, to set the install path to ``/usr/local``, use Armadillo as the linear algebra library, and enable OpenMP features, we would run:

.. code:: bash

    ./configure -i "/usr/local" -l arma -p

Following this with the standard ``make && make install`` would build the library and install into ``/usr/local``.

----

Installation Method 2: Header-only Library
------------------------------------------

OptimLib is also available as a header-only library (i.e., without the need to compile a shared library). Simply run ``configure`` with the ``--header-only-version`` option:

.. code:: bash

    ./configure --header-only-version

This will create a new directory, ``header_only_version``, containing a copy of OptimLib, modified to work on an inline basis. 
With this header-only version, simply include the header files (``#include "optim.hpp``) and set the include path to the ``head_only_version`` directory (e.g.,``-I/path/to/optimlib/header_only_version``).
