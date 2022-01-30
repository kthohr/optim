# OptimLib &nbsp; [![Build Status](https://github.com/kthohr/optim/actions/workflows/main.yml/badge.svg)](https://github.com/kthohr/optim/actions/workflows/main.yml) [![Coverage Status](https://codecov.io/github/kthohr/optim/coverage.svg?branch=master)](https://codecov.io/github/kthohr/optim?branch=master) [![License](https://img.shields.io/badge/Licence-Apache%202.0-blue.svg)](./LICENSE) [![Documentation Status](https://readthedocs.org/projects/optimlib/badge/?version=latest)](https://optimlib.readthedocs.io/en/latest/?badge=latest)

OptimLib is a lightweight C++ library of numerical optimization methods for nonlinear functions.

Features:

* A C++11/14/17 library of local and global optimization algorithms, as well as root finding techniques.
* Derivative-free optimization using advanced, parallelized metaheuristic methods.
* Constrained optimization routines to handle simple box constraints, as well as systems of nonlinear constraints.
* For fast and efficient matrix-based computation, OptimLib supports the following templated linear algebra libraries:
  * [Armadillo](http://arma.sourceforge.net/)
  * [Eigen](http://eigen.tuxfamily.org/index.php) (version >= 3.4.0)
* Automatic differentiation functionality is available through use of the [Autodiff library](https://autodiff.github.io)
* OpenMP-accelerated algorithms for parallel computation. 
* Straightforward linking with parallelized BLAS libraries, such as [OpenBLAS](https://github.com/xianyi/OpenBLAS).
* Available as a single precision (``float``) or double precision (``double``) library.
* Available as a header-only library, or as a compiled shared library.
* Released under a permissive, non-GPL license.

### Contents:
* [Algorithms](#algorithms)
* [Documentation](#documentation)
* [General API](#api)
* [Installation](#installation)
* [R Compatibility](#r-compatibility)
* [Examples](#examples)
* [Automatic Differentiation](#automatic-differentiation)
* [Author and License](#author)

## Algorithms

A list of currently available algorithms includes:

* Broyden's Method (for root finding)
* Newton's method, BFGS, and L-BFGS
* Gradient descent: basic, momentum, Adam, AdaMax, Nadam, NadaMax, and more
* Nonlinear Conjugate Gradient
* Nelder-Mead
* Differential Evolution (DE)
* Particle Swarm Optimization (PSO)

## Documentation

Full documentation is available online:

[![Documentation Status](https://readthedocs.org/projects/optimlib/badge/?version=latest)](https://optimlib.readthedocs.io/en/latest/?badge=latest)

A PDF version of the documentation is available [here](https://buildmedia.readthedocs.org/media/pdf/optimlib/latest/optimlib.pdf).

## API

The OptimLib API follows a relatively simple convention, with most algorithms called in the following manner:
```
algorithm_id(<initial/final values>, <objective function>, <objective function data>);
```
The inputs, in order, are:
* A writable vector of initial values to define the starting point of the algorithm. In the event of successful completion, the initial values will be overwritten by the solution vector.
* The 'objective function' is the user-defined function to be minimized (or zeroed-out in the case of root finding methods).
* The final input is optional: it is any object that contains additional parameters necessary to evaluate the objective function.

For example, the BFGS algorithm is called using
```cpp
bfgs(ColVec_t& init_out_vals, std::function<double (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, void* opt_data);
```

where ``ColVec_t`` is used to represent, e.g., ``arma::vec`` or ``Eigen::VectorXd`` types.

## Installation

OptimLib is available as a compiled shared library, or as header-only library, for Unix-alike systems only (e.g., popular Linux-based distros, as well as macOS). Use of this library with Windows-based systems, with or without MSVC, **is not supported**.

### Requirements

OptimLib requires either the Armadillo or Eigen C++ linear algebra libraries. (Note that Eigen version 3.4.0 requires a C++14-compatible compiler.)

Before including the header files, define **one** of the following:
``` cpp
#define OPTIM_ENABLE_ARMA_WRAPPERS
#define OPTIM_ENABLE_EIGEN_WRAPPERS
```

Example:
``` cpp
#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include "optim.hpp"
```

### Installation Method 1: Shared Library

The library can be installed on Unix-alike systems via the standard `./configure && make` method.

First clone the library and any necessary submodules:

``` bash
# clone optim into the current directory
git clone https://github.com/kthohr/optim ./optim

# change directory
cd ./optim

# clone necessary submodules
git submodule update --init
```

Set (one) of the following environment variables *before* running `configure`:

``` bash
export ARMA_INCLUDE_PATH=/path/to/armadillo
export EIGEN_INCLUDE_PATH=/path/to/eigen
```

Finally:

``` bash
# build and install with Eigen
./configure -i "/usr/local" -l eigen -p
make
make install
```

The final command will install OptimLib into `/usr/local`.

Configuration options (see `./configure -h`):

&nbsp; &nbsp; &nbsp; **Primary**
* `-h` print help
* `-i` installation path; default: the build directory
* `-f` floating-point precision mode; default: `double`
* `-l` specify the choice of linear algebra library; choose `arma` or `eigen`
* `-m` specify the BLAS and Lapack libraries to link with; for example, `-m "-lopenblas"` or `-m "-framework Accelerate"`
* `-o` compiler optimization options; defaults to `-O3 -march=native -ffp-contract=fast -flto -DARMA_NO_DEBUG`
* `-p` enable OpenMP parallelization features (*recommended*)

&nbsp; &nbsp; &nbsp; **Secondary**
* `-c` a coverage build (used with Codecov)
* `-d` a 'development' build
* `-g` a debugging build (optimization flags set to `-O0 -g`)

&nbsp; &nbsp; &nbsp; **Special**
* `--header-only-version` generate a header-only version of OptimLib (see [below](#installation-method-2-header-only-library))
<!-- * `-R` RcppArmadillo compatible build by setting the appropriate R library directories (R, Rcpp, and RcppArmadillo) -->

## Installation Method 2: Header-only Library

OptimLib is also available as a header-only library (i.e., without the need to compile a shared library). Simply run `configure` with the `--header-only-version` option:

```bash
./configure --header-only-version
```

This will create a new directory, `header_only_version`, containing a copy of OptimLib, modified to work on an inline basis. With this header-only version, simply include the header files (`#include "optim.hpp`) and set the include path to the `head_only_version` directory (e.g.,`-I/path/to/optimlib/header_only_version`).

## R Compatibility

To use OptimLib with an R package, first generate a header-only version of the library (see [above](#installation-method-2-header-only-library)). Then simply add a compiler definition before including the OptimLib files.

* For RcppArmadillo:
```cpp
#define OPTIM_USE_RCPP_ARMADILLO
#include "optim.hpp"
```

* For RcppEigen:
```cpp
#define OPTIM_USE_RCPP_EIGEN
#include "optim.hpp"
```

## Examples

To illustrate OptimLib at work, consider searching for the global minimum of the [Ackley function](https://en.wikipedia.org/wiki/Ackley_function):

![Ackley](https://github.com/kthohr/kthohr.github.io/blob/master/pics/ackley_fn_3d.png)

This is a well-known test function with many local minima. Newton-type methods (such as BFGS) are sensitive to the choice of initial values, and will perform rather poorly here. As such, we will employ a global search method; in this case: Differential Evolution.

Code:

``` cpp
#define OPTIM_ENABLE_ARMA_WRAPPERS
#include "optim.hpp"

//
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
```

Compile and run:

``` bash
g++ -Wall -std=c++11 -O3 -march=native -ffp-contract=fast -I/path/to/armadillo -I/path/to/optim/include optim_de_ex.cpp -o optim_de_ex.out -L/path/to/optim/lib -loptim
./optim_de_ex.out
```

Output:
```
de: Ackley test completed successfully.
elapsed time: 0.028167s

de: solution to Ackley test:
  -1.2702e-17
  -3.8432e-16
```
On a standard laptop, OptimLib will compute a solution to within machine precision in a fraction of a second.

Check the `/tests` directory for additional examples, and http://www.kthohr.com/optimlib.html for a detailed description of each algorithm.

### Logistic regression

For a data-based example, consider maximum likelihood estimation of a logit model, common in statistics and machine learning. In this case we have closed-form expressions for the gradient and hessian. We will employ a popular gradient descent method, Adam (Adaptive Moment Estimation), and compare to a pure Newton-based algorithm.

``` cpp
#define OPTIM_ENABLE_ARMA_WRAPPERS
#include "optim.hpp"

// sigmoid function

inline
arma::mat sigm(const arma::mat& X)
{
    return 1.0 / (1.0 + arma::exp(-X));
}

// log-likelihood function data

struct ll_data_t
{
    arma::vec Y;
    arma::mat X;
};

// log-likelihood function with hessian

double ll_fn_whess(const arma::vec& vals_inp, arma::vec* grad_out, arma::mat* hess_out, void* opt_data)
{
    ll_data_t* objfn_data = reinterpret_cast<ll_data_t*>(opt_data);

    arma::vec Y = objfn_data->Y;
    arma::mat X = objfn_data->X;

    arma::vec mu = sigm(X*vals_inp);

    const double norm_term = static_cast<double>(Y.n_elem);

    const double obj_val = - arma::accu( Y%arma::log(mu) + (1.0-Y)%arma::log(1.0-mu) ) / norm_term;

    //

    if (grad_out)
    {
        *grad_out = X.t() * (mu - Y) / norm_term;
    }

    //

    if (hess_out)
    {
        arma::mat S = arma::diagmat( mu%(1.0-mu) );
        *hess_out = X.t() * S * X / norm_term;
    }

    //

    return obj_val;
}

// log-likelihood function for Adam

double ll_fn(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    return ll_fn_whess(vals_inp,grad_out,nullptr,opt_data);
}

//

int main()
{
    int n_dim = 5;     // dimension of parameter vector
    int n_samp = 4000; // sample length

    arma::mat X = arma::randn(n_samp,n_dim);
    arma::vec theta_0 = 1.0 + 3.0*arma::randu(n_dim,1);

    arma::vec mu = sigm(X*theta_0);

    arma::vec Y(n_samp);

    for (int i=0; i < n_samp; i++)
    {
        Y(i) = ( arma::as_scalar(arma::randu(1)) < mu(i) ) ? 1.0 : 0.0;
    }

    // fn data and initial values

    ll_data_t opt_data;
    opt_data.Y = std::move(Y);
    opt_data.X = std::move(X);

    arma::vec x = arma::ones(n_dim,1) + 1.0; // initial values

    // run Adam-based optim

    optim::algo_settings_t settings;

    settings.gd_method = 6;
    settings.gd_settings.step_size = 0.1;

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

    bool success = optim::gd(x,ll_fn,&opt_data,settings);

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    //

    if (success) {
        std::cout << "Adam: logit_reg test completed successfully.\n"
                  << "elapsed time: " << elapsed_seconds.count() << "s\n";
    } else {
        std::cout << "Adam: logit_reg test completed unsuccessfully." << std::endl;
    }

    arma::cout << "\nAdam: true values vs estimates:\n" << arma::join_rows(theta_0,x) << arma::endl;

    //
    // run Newton-based optim

    x = arma::ones(n_dim,1) + 1.0; // initial values

    start = std::chrono::system_clock::now();

    success = optim::newton(x,ll_fn_whess,&opt_data);

    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;

    //

    if (success) {
        std::cout << "newton: logit_reg test completed successfully.\n"
                  << "elapsed time: " << elapsed_seconds.count() << "s\n";
    } else {
        std::cout << "newton: logit_reg test completed unsuccessfully." << std::endl;
    }

    arma::cout << "\nnewton: true values vs estimates:\n" << arma::join_rows(theta_0,x) << arma::endl;

    return 0;
}
```
Output:
```
Adam: logit_reg test completed successfully.
elapsed time: 0.025128s

Adam: true values vs estimates:
   2.7850   2.6993
   3.6561   3.6798
   2.3379   2.3860
   2.3167   2.4313
   2.2465   2.3064

newton: logit_reg test completed successfully.
elapsed time: 0.255909s

newton: true values vs estimates:
   2.7850   2.6993
   3.6561   3.6798
   2.3379   2.3860
   2.3167   2.4313
   2.2465   2.3064
```

## Automatic Differentiation

By combining Eigen with the [Autodiff library](https://autodiff.github.io), OptimLib provides experimental support for automatic differentiation. See the [documentation](https://optimlib.readthedocs.io/en/latest/autodiff.html) for more details on this topic.

## Author

Keith O'Hara

## License

Apache Version 2
