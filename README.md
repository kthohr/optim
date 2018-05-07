# OptimLib &nbsp; [![Build Status](https://travis-ci.org/kthohr/optim.svg?branch=master)](https://travis-ci.org/kthohr/optim) [![Coverage Status](https://codecov.io/github/kthohr/optim/coverage.svg?branch=master)](https://codecov.io/github/kthohr/optim?branch=master) [![License](https://img.shields.io/badge/Licence-Apache%202.0-blue.svg)](./LICENSE)

OptimLib is a lightweight C++ library of numerical optimization methods for nonlinear functions.

Features:

* Parallelized C++11 library of local and global optimization algorithms, as well as root finding techniques.
* Numerous derivative-free algorithms including advanced metaheuristics.
* Constrained optimization routines can handle simple box constraints or systems of complicated nonlinear constraints.
* Built on the [Armadillo C++ linear algebra library](http://arma.sourceforge.net/) for fast and efficient matrix-based computation.
* Released under a permissive, non-GPL license.

## Status

The library is actively maintained, and is still being extended. A list of algorithms includes:

* Newton's method, BFGS, and L-BFGS
* Nonlinear Conjugate Gradient
* Broyden's Method
* Nelder-Mead
* Differential Evolution (DE)
* Particle Swarm Optimization (PSO)

## Syntax

OptimLib functions generally exhibit the following structure:
```
algorithm(<initial and output values>, <objective function>, <data for the objective function>)
```
* The vector of initial values defines the starting point for the algorithm, and will contain the solution vector upon successful completion of the algorithm.
* The 'objective function' is the function to be minimized over, or zeroed-out in the case of Broyden's method.
* The final input passes any additional parameters required to evaluate the objective function (*optional*).

For example, the BFGS algorithm is called using:
``` cpp
bool bfgs(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data);
```

## Installation

The library is installed in the usual way:

```bash
# clone optim into the current directory
git clone https://github.com/kthohr/optim ./optim
# build and install
cd ./optim
./configure -i "/usr/local" -p
make
make install
```

The last line will install OptimLib into `/usr/local`.

There are several configuration options available (see `./configure -h`):
* `-c` a coverage build (used with Codecov)
* `-d` a 'development' build
* `-g` a debugging build (optimization flags set to `-O0 -g`)
* `-h` print help
* `-i` install path; default: build directory
* `-m` specify the BLAS and Lapack libraries to link against; for example, `-m "-lopenblas"` or `-m "-framework Accelerate"`
* `-o` compiler optimization options; defaults to `-O3 -march=native -ffp-contract=fast -flto -DARMA_NO_DEBUG`
* `-p` enable OpenMP parallelization features (*recommended*)

### Armadillo

OptimLib is built on the Armadillo C++ linear algebra library. The `configure` script will search for Armadillo files in the usual places: `/usr/include`, `/usr/local/include`, `/opt/include`, `/opt/local/include`. If the Armadillo header files are installed elsewhere, set:
``` bash
export ARMA_INCLUDE_PATH=/path/to/armadillo
```
Set this environment variable before before running `./configure`. Otherwise the build script will download the required files from the Armadillo GitHub repository.

## Example

Find the global minimum of the [Ackley function](https://en.wikipedia.org/wiki/Ackley_function):

![Ackley](https://github.com/kthohr/kthohr.github.io/blob/master/pics/ackley_fn_3d.png)

This is a well-known test function that contains many local minima. Newton-type methods (like BFGS) are sensitive to the choice of initial values, and will perform rather poorly here. As such, we will employ a global search method: Differential Evolution.

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

Output:
```
de: Ackley test completed successfully.
elapsed time: 0.028167s

de: solution to Ackley test:
  -1.2702e-17
  -3.8432e-16
```
On a standard laptop, OptimLib will compute the solution to within machine precision in a fraction of a second.

See http://www.kthohr.com/optimlib.html for a detailed description of each algorithm, and more examples.

## Author

Keith O'Hara

## License

Apache Version 2
