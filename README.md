# OptimLib &nbsp; [![Build Status](https://travis-ci.org/kthohr/optim.svg?branch=master)](https://travis-ci.org/kthohr/optim) [![Coverage Status](https://codecov.io/github/kthohr/optim/coverage.svg?branch=master)](https://codecov.io/github/kthohr/optim?branch=master)

## About

OptimLib is a lightweight C++ library for numerical optimization of nonlinear functions.

* Parallelized C++11 library of local and global optimization methods, as well as root finding techniques.
* Numerous derivative-free algorithms including including advanced and hybrid metaheuristics.
* Constrained optimization: simple box constraints or complicated nonlinear constraints.
* Built on the Armadillo C++ linear algebra library for fast and efficient matrix-based computation.

## Status

The library is actively maintained, and is still being extended.

A list of features includes:

* BFGS and L-BFGS
* Nonlinear Conjugate Gradient
* Broyden's Method
* Differential Evolution (DE)
* Particle Swarm Optimization (PSO)

## Syntax

OptimLib functions are generally defined loosely as
```
algorithm(<initial and end values>, <objective function>, <data for objective function>)
```
where the inputs, in order, are:
* initial values define the starting point for the algorithm, and will also contain the solution vector;
* the objective function to be minimized; and
* additional parameters passed to the objective function.

For example, the BFGS algorithm is called using:
``` cpp
bool bfgs(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data);
```

## Installation

The library is installed in the usual way:

```bash
# clone optim
git clone -b master --single-branch https://github.com/kthohr/optim ./optim
# build and install
cd ./optim
./configure
make
make install
```

The last line will install OptimLib into /usr/local

There are several configure options available:
* ```-b``` dev a 'development' build with install names set to the build directory (as opposed to an install path)
* ```-c``` a coverage build
* ```-m``` specify the BLAS and Lapack libraries to link against; for example, ```-m "-lopenblas"``` or ```-m "-framework Accelerate"```
* ```-o``` enable aggressive compiler optimization features: ```-o fast``` or ```-o native```
* ```-p``` enable parallelization features (using OpenMP)


## Example

![Ackley](https://github.com/kthohr/kthohr.github.io/blob/master/pics/ackley_fn_3d.png)

See http://www.kthohr.com/optimlib.html for details and more examples.

## Author

Keith O'Hara

## License

GPL (>= 2)

