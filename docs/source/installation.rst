.. Copyright (c) 2016-2020 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

.. _installation:

Installation
============

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


In addition, OptimLib requires either the Armadillo or Eigen C++ linear algebra libraries. Set (one) of the following environment variables *before* running `configure`:

.. code:: bash
    
    export ARMA_INCLUDE_PATH=/path/to/armadillo
    export EIGEN_INCLUDE_PATH=/path/to/eigen

For example, to set the install path to ``/usr/local``, use Armadillo as the linear algebra library, and enable OpenMP features, we would use:

.. code:: bash

    ./configure -i "/usr/local" -l arma -p
    make

----

The following options should be declared **before** including the OptimLib header files. 

- OpenMP functionality is enabled by default if the ``_OPENMP`` macro is detected (e.g., by invoking ``-fopenmp`` with GCC or Clang). To explicitly enable OpenMP features use:

.. code:: cpp

    #define OPTIM_USE_OPENMP

- To disable OpenMP functionality:

.. code:: cpp

    #define OPTIM_DONT_USE_OPENMP

- To use OptimLib with Armadillo or Eigen:

.. code:: cpp

    #define OPTIM_ENABLE_ARMA_WRAPPERS
    #define OPTIM_ENABLE_EIGEN_WRAPPERS

