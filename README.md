# OptimLib &nbsp; [![Build Status](https://travis-ci.org/kthohr/optim.svg?branch=master)](https://travis-ci.org/kthohr/optim) [![Coverage Status](https://codecov.io/github/kthohr/optim/coverage.svg?branch=master)](https://codecov.io/github/kthohr/optim?branch=master) [![License](https://img.shields.io/badge/Licence-Apache%202.0-blue.svg)](./LICENSE)

OptimLib is a lightweight C++ library of numerical optimization methods for nonlinear functions.

Features:

* C++11 library of local and global optimization algorithms, as well as root finding techniques.
* Derivative-free optimization using advanced, parallelized metaheuristics.
* Constrained optimization routines can handle simple box constraints, or systems of nonlinear constraints.
* Built on the [Armadillo C++ linear algebra library](http://arma.sourceforge.net/) for fast and efficient matrix-based computation.
* OpenMP-accelerated library, and straightforward linking with parallelized BLAS libraries such as [OpenBLAS](https://github.com/xianyi/OpenBLAS).
* Released under a permissive, non-GPL license.

### Contents:
* [Status](#status) 
* [Syntax](#syntax)
* [Installation](#installation)
* [Examples](#examples)
* [License](#license)

## Status

The library is actively maintained, and is still being extended. A list of algorithms includes:

* Broyden's Method (for root finding)
* Newton's method, BFGS, and L-BFGS
* Gradient descent: basic, momentum, Adam, and more
* Nonlinear Conjugate Gradient
* Nelder-Mead
* Differential Evolution (DE)
* Particle Swarm Optimization (PSO)

## Syntax

OptimLib functions generally have the following structure:
```
algorithm(<initial-to-output values>, <objective function>, <objective function data>)
```
In order:
* A writable vector of initial values that defines the starting point of the algorithm, and will contain the solution vector upon successful completion of the algorithm.
* The 'objective function' is the function to be minimized, or zeroed-out in the case of root finding.
* The final input is optional: it is any object that contains additional parameters required to evaluate the objective function.

For example, the BFGS algorithm is called using
``` cpp
bool bfgs(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data);
```

## Installation

The library is installed on Unix-alike systems via the standard `./configure && make` method:

```bash
# clone optim into the current directory
git clone https://github.com/kthohr/optim ./optim
# build and install
cd ./optim
./configure -i "/usr/local" -p
make
make install
```

The final command will install OptimLib into `/usr/local`.

There are several configuration options available (see `./configure -h`):
* `-c` a coverage build (used with Codecov)
* `-d` a 'development' build
* `-g` a debugging build (optimization flags set to `-O0 -g`)
* `-h` print help
* `-i` install path; default: the build directory
* `-m` specify the BLAS and Lapack libraries to link against; for example, `-m "-lopenblas"` or `-m "-framework Accelerate"`
* `-o` compiler optimization options; defaults to `-O3 -march=native -ffp-contract=fast -flto -DARMA_NO_DEBUG`
* `-p` enable OpenMP parallelization features (*recommended*)

### Armadillo

OptimLib is built on the Armadillo C++ linear algebra library. The `configure` script will search for Armadillo files in the usual places: `/usr/include`, `/usr/local/include`, `/opt/include`, `/opt/local/include`. If the Armadillo header files are installed elsewhere, set the following environment variable *before* running `configure`:
``` bash
export ARMA_INCLUDE_PATH=/path/to/armadillo
```
Otherwise the build script will download the required files from the Armadillo GitHub repository.

## Examples

To illustrate OptimLib at work, consider searching for the global minimum of the [Ackley function](https://en.wikipedia.org/wiki/Ackley_function):

![Ackley](https://github.com/kthohr/kthohr.github.io/blob/master/pics/ackley_fn_3d.png)

This is a well-known test function with many local minima. Newton-type methods (such as BFGS) are sensitive to the choice of initial values, and will perform rather poorly here. As such, we will employ a global search method; in this case: Differential Evolution.

Code:

``` cpp
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

## Author

Keith O'Hara

## License

Apache Version 2
